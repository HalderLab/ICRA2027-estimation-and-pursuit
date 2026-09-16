classdef IMEBlock < matlab.System
%IMEBLOCK Internal Model Estimator for Simulink.
%
% Inputs:
%   t               scalar simulation time [s]
%   y               [rho; alpha2]
%   eta2            [v2; u2]
%   newMeasurement  logical / 0-1 flag
%
% Outputs:
%   zHat             [xHat; yHat; thetaHat]
%   eta1Hat          [v1Hat; u1Hat]
%   ready            double 0/1
%   cost             latest IME optimization cost
%   iterations       forward-backward iterations at this call
%
% Augmented internal state:
%   xiHat = [xHat; yHat; thetaHat; v1Hat; u1Hat]
%
% Optimization control:
%   omegaHat = [d(v1Hat)/dt; d(u1Hat)/dt]
%
% There is no arrival cost. Previous moving-window solutions are used
% only as warm starts. Cold-start hidden-state guesses can be configured
% before the first step() call using setInitialGuess().

    % Initial guesses are exposed as nontunable System-object properties so
    % the same IME implementation can be used both in Simulink and in the
    % Monte Carlo scripts.  They must be set before the first call to step().
    properties (Nontunable)
        InitialThetaGuess = 0.20
        InitialV1Guess = 1.00
        InitialU1Guess = 0.00
    end

    properties (Constant, Access = private)
        Nh = 30
        MaxNodes = 31
        NSub = 5
        MaxM = 150
        MaxFineNodes = 151
        MinSamples = 5

        MaxControlSamples = 10000

        Qx = 20.0
        Qy = 20.0

        ChiV = 2.0
        ChiU = 2.0


        MaxIterBoot = 120
        MaxIterRT = 10
        RelCostTol = 1e-6
        ArmijoC = 1e-4
        MaxLineSearch = 20

        % Numerical scaling only; not part of the objective.
        Z0Scale = [ ...
            0.05^2; ...
            0.05^2; ...
            (30*pi/180)^2; ...
            0.40^2; ...
            0.40^2]

        PredictionMaxStep = 0.01
    end

    properties (Access = private)
        % LiDAR history
        tBuf
        rhoBuf
        alpha2Buf
        nMeas

        % High-rate known eta2 history (circular buffer)
        controlTBuf
        controlVBuf
        controlUBuf
        nControl
        controlStart

        % Previous optimized trajectory
        prevZ
        prevW
        prevM
        prevValid

        % Prediction state
        zPred
        tPred
        vSelfPred
        uSelfPred
        predictorValid

        % Most recent estimate/diagnostics
        lastAug
        lastReady
        lastCost
        lastMeasurementTime
    end

    methods
        function setInitialGuess(obj,thetaGuess,v1Guess,u1Guess)
            %SETINITIALGUESS Configure the cold-start hidden-state guess.
            % Must be called before the first step() call.  The default
            % values remain [0.20 rad, 1.00 m/s, 0.00 rad/s] for Simulink
            % models that do not explicitly override them.

            vals = [thetaGuess,v1Guess,u1Guess];
            if any(~isfinite(vals))
                error('Initial guesses must be finite.');
            end

            if isLocked(obj)
                error('setInitialGuess() must be called before the first step() call or after release().');
            end

            obj.InitialThetaGuess = atan2(sin(thetaGuess),cos(thetaGuess));
            obj.InitialV1Guess = max(v1Guess,0.0);
            obj.InitialU1Guess = u1Guess;
        end
    end

    methods (Access = protected)

        function setupImpl(obj)
            obj.initializeState();
        end

        function resetImpl(obj)
            obj.initializeState();
        end

        function [zHat,eta1Hat,ready,cost,iterations] = ...
                stepImpl(obj,t,y,eta2,newMeasurement)

            rho = y(1);
            alpha2 = y(2);

            v2 = eta2(1);
            u2 = eta2(2);

            % Record known self-motion at every block execution.
            obj.recordControlSample(t,v2,u2);

            if newMeasurement ~= 0
                [zAug,isReady,costValue,iterValue] = ...
                    obj.measurementUpdate(t,rho,alpha2,v2,u2);
            else
                [zAug,isReady,costValue,iterValue] = ...
                    obj.predictionUpdate(t,v2,u2);
            end

            zHat = zAug(1:3);
            eta1Hat = zAug(4:5);

            % Resolve the equivalent (theta + pi, -v1) representation.
            % The physical convention used by the benchmark defines v1 as
            % nonnegative forward speed. This changes only the reported
            % representation, not the optimized IME trajectory or objective.
            if eta1Hat(1) < 0
                eta1Hat(1) = -eta1Hat(1);
                zHat(3) = zHat(3) + pi;
            end

            zHat(3) = obj.wrapPi(zHat(3));

            % Double-valued ready avoids type conflicts in Simulink Muxes.
            ready = double(isReady);
            cost = costValue;
            iterations = double(iterValue);
        end

        function [s1,s2,s3,s4,s5] = getOutputSizeImpl(~)
            s1 = [3 1];
            s2 = [2 1];
            s3 = [1 1];
            s4 = [1 1];
            s5 = [1 1];
        end

        function [d1,d2,d3,d4,d5] = getOutputDataTypeImpl(~)
            d1 = 'double';
            d2 = 'double';
            d3 = 'double';
            d4 = 'double';
            d5 = 'double';
        end

        function [c1,c2,c3,c4,c5] = isOutputComplexImpl(~)
            c1 = false;
            c2 = false;
            c3 = false;
            c4 = false;
            c5 = false;
        end

        function [f1,f2,f3,f4,f5] = isOutputFixedSizeImpl(~)
            f1 = true;
            f2 = true;
            f3 = true;
            f4 = true;
            f5 = true;
        end

        function [n1,n2,n3,n4] = getInputNamesImpl(~)
            n1 = 't';
            n2 = 'y';
            n3 = 'eta2';
            n4 = 'newMeasurement';
        end

        function [n1,n2,n3,n4,n5] = getOutputNamesImpl(~)
            n1 = 'zHat';
            n2 = 'eta1Hat';
            n3 = 'ready';
            n4 = 'cost';
            n5 = 'iterations';
        end
    end

    methods (Access = private)

        function initializeState(obj)
            obj.tBuf = zeros(1,obj.MaxNodes);
            obj.rhoBuf = zeros(1,obj.MaxNodes);
            obj.alpha2Buf = zeros(1,obj.MaxNodes);
            obj.nMeas = 0;

            obj.controlTBuf = zeros(1,obj.MaxControlSamples);
            obj.controlVBuf = zeros(1,obj.MaxControlSamples);
            obj.controlUBuf = zeros(1,obj.MaxControlSamples);
            obj.nControl = 0;
            obj.controlStart = 1;

            obj.prevZ = zeros(5,obj.MaxFineNodes);
            obj.prevW = zeros(2,obj.MaxM);
            obj.prevM = 0;
            obj.prevValid = false;

            obj.zPred = zeros(5,1);
            obj.tPred = 0.0;
            obj.vSelfPred = 0.0;
            obj.uSelfPred = 0.0;
            obj.predictorValid = false;

            obj.lastAug = zeros(5,1);
            obj.lastReady = false;
            obj.lastCost = NaN;
            obj.lastMeasurementTime = NaN;
        end

        function [zAug,isReady,costValue,iterValue] = ...
                measurementUpdate(obj,t,rho,alpha2,v2,u2)

            rho = max(rho,1e-6);
            alpha2 = obj.wrapPi(alpha2);

            windowShifted = false;

            if obj.nMeas < obj.MaxNodes
                obj.nMeas = obj.nMeas + 1;
                idx = obj.nMeas;
            else
                for k = 1:obj.MaxNodes-1
                    obj.tBuf(k) = obj.tBuf(k+1);
                    obj.rhoBuf(k) = obj.rhoBuf(k+1);
                    obj.alpha2Buf(k) = obj.alpha2Buf(k+1);
                end

                idx = obj.MaxNodes;
                windowShifted = true;
            end

            obj.tBuf(idx) = t;
            obj.rhoBuf(idx) = rho;
            obj.alpha2Buf(idx) = alpha2;
            obj.lastMeasurementTime = t;

            if obj.nMeas < obj.MinSamples
                zAug = [ ...
                    rho*cos(alpha2); ...
                    rho*sin(alpha2); ...
                    obj.InitialThetaGuess; ...
                    obj.InitialV1Guess; ...
                    obj.InitialU1Guess];

                isReady = false;
                costValue = NaN;
                iterValue = 0;

                obj.lastAug = zAug;
                obj.lastReady = false;
                obj.lastCost = NaN;
                obj.predictorValid = false;
                return;
            end

            Y = zeros(2,obj.MaxNodes);

            for k = 1:obj.nMeas
                Y(1,k) = obj.rhoBuf(k)*cos(obj.alpha2Buf(k));
                Y(2,k) = obj.rhoBuf(k)*sin(obj.alpha2Buf(k));
            end

            currentM = (obj.nMeas-1)*obj.NSub;
            W = zeros(2,obj.MaxM);

            if ~obj.prevValid
                % Cold start: guesses are initialization only.
                z0 = [ ...
                    Y(1,1); ...
                    Y(2,1); ...
                    obj.InitialThetaGuess; ...
                    obj.InitialV1Guess; ...
                    obj.InitialU1Guess];

                maxIter = obj.MaxIterBoot;

            elseif ~windowShifted
                % Growing horizon: keep the previous left edge and complete
                % overlapping omega trajectory.
                z0 = obj.prevZ(:,1);

                nCopy = min(obj.prevM,currentM);

                for j = 1:nCopy
                    W(:,j) = obj.prevW(:,j);
                end

                if currentM > obj.prevM
                    if obj.prevM > 0
                        wEnd = obj.prevW(:,obj.prevM);
                    else
                        wEnd = zeros(2,1);
                    end

                    for j = obj.prevM+1:currentM
                        W(:,j) = wEnd;
                    end
                end

                maxIter = obj.MaxIterRT;

            else
                % Receding horizon: shift the previous trajectory by one
                % LiDAR interval.
                shiftNode = obj.NSub + 1;
                z0 = obj.prevZ(:,shiftNode);

                overlapM = obj.prevM - obj.NSub;
                if overlapM < 0
                    overlapM = 0;
                end

                for j = 1:overlapM
                    W(:,j) = obj.prevW(:,j+obj.NSub);
                end

                if obj.prevM > 0
                    wEnd = obj.prevW(:,obj.prevM);
                else
                    wEnd = zeros(2,1);
                end

                for j = overlapM+1:currentM
                    W(:,j) = wEnd;
                end

                maxIter = obj.MaxIterRT;
            end

            [W,Z,costValue,iterValue] = ...
                obj.solveWindow(z0,W,currentM,Y,obj.nMeas,maxIter);

            obj.prevZ(:) = 0;
            obj.prevW(:) = 0;

            for j = 1:currentM+1
                obj.prevZ(:,j) = Z(:,j);
            end

            for j = 1:currentM
                obj.prevW(:,j) = W(:,j);
            end

            obj.prevM = currentM;
            obj.prevValid = true;

            zAug = Z(:,currentM+1);

            obj.zPred = zAug;
            obj.tPred = t;
            obj.vSelfPred = v2;
            obj.uSelfPred = u2;
            obj.predictorValid = true;

            isReady = true;

            obj.lastAug = zAug;
            obj.lastReady = true;
            obj.lastCost = costValue;
        end

        function [zAug,isReady,costValue,iterValue] = ...
                predictionUpdate(obj,t,v2,u2)

            iterValue = 0;

            if ~obj.predictorValid
                zAug = obj.lastAug;
                isReady = obj.lastReady;
                costValue = obj.lastCost;
                return;
            end

            dt = t-obj.tPred;

            if dt > 0
                obj.zPred = obj.propagatePrediction( ...
                    obj.zPred,obj.tPred,t, ...
                    obj.vSelfPred,obj.uSelfPred,v2,u2);

                obj.tPred = t;
            end

            obj.vSelfPred = v2;
            obj.uSelfPred = u2;

            zAug = obj.zPred;
            isReady = obj.lastReady;
            costValue = obj.lastCost;
            obj.lastAug = zAug;
        end

        function [W,Z,J,iter] = ...
                solveWindow(obj,z0,W,M,Y,nMeas,maxIter)

            tFine = obj.makeFineGrid(nMeas);

            [vL,vM,vR,uL,uM,uR] = ...
                obj.prepareKnownControls(tFine,M);

            [J,Z] = obj.forwardCost( ...
                z0,W,M,Y,nMeas,tFine, ...
                vL,vM,vR,uL,uM,uR);

            iter = 0;

            for it = 1:obj.MaxIterBoot
                if it > maxIter
                    break;
                end

                iter = it;

                [g0,gW] = obj.backwardSweep( ...
                    W,Z,M,Y,nMeas,tFine,uL,uM,uR);

                % Initial augmented-state descent direction. There is no
                % arrival-cost contribution to g0.
                dz = -obj.Z0Scale.*g0;

                dW = zeros(2,obj.MaxM);

                for j = 1:M
                    dtj = tFine(j+1)-tFine(j);

                    dW(1,j) = ...
                        -gW(1,j)/(dtj*obj.ChiV);

                    dW(2,j) = ...
                        -gW(2,j)/(dtj*obj.ChiU);
                end

                directionalDerivative = g0.'*dz;

                for j = 1:M
                    directionalDerivative = directionalDerivative + ...
                        gW(1,j)*dW(1,j) + ...
                        gW(2,j)*dW(2,j);
                end

                if directionalDerivative >= -1e-14
                    break;
                end

                step = 1.0;
                accepted = false;

                JTry = J;
                ZTry = Z;
                WTry = W;
                zTry = z0;

                for ls = 1:obj.MaxLineSearch
                    zTry = z0 + step*dz;
                    WTry = W;

                    for j = 1:M
                        WTry(:,j) = W(:,j) + step*dW(:,j);
                    end

                    [JTry,ZTry] = obj.forwardCost( ...
                        zTry,WTry,M,Y,nMeas,tFine, ...
                        vL,vM,vR,uL,uM,uR);

                    if JTry <= ...
                            J + obj.ArmijoC*step*directionalDerivative
                        accepted = true;
                        break;
                    end

                    step = 0.5*step;
                end

                if ~accepted
                    break;
                end

                relDecrease = abs(J-JTry)/max(1.0,abs(J));

                z0 = zTry;
                W = WTry;
                Z = ZTry;
                J = JTry;

                if relDecrease < obj.RelCostTol
                    break;
                end
            end
        end

        function [J,Z] = forwardCost( ...
                obj,z0,W,M,Y,nMeas,tFine, ...
                vL,vM,vR,uL,uM,uR)

            Z = zeros(5,obj.MaxFineNodes);
            Z(:,1) = z0;

            for j = 1:M
                h = tFine(j+1)-tFine(j);
                w = W(:,j);

                k1 = obj.relativeDynamics( ...
                    Z(:,j),w,vL(j),uL(j));

                k2 = obj.relativeDynamics( ...
                    Z(:,j)+0.5*h*k1,w,vM(j),uM(j));

                k3 = obj.relativeDynamics( ...
                    Z(:,j)+0.5*h*k2,w,vM(j),uM(j));

                k4 = obj.relativeDynamics( ...
                    Z(:,j)+h*k3,w,vR(j),uR(j));

                Z(:,j+1) = ...
                    Z(:,j) + (h/6)*(k1+2*k2+2*k3+k4);
            end

            % No arrival cost.
            J = 0.0;

            for k = 1:nMeas
                node = 1 + (k-1)*obj.NSub;

                ex = Z(1,node)-Y(1,k);
                ey = Z(2,node)-Y(2,k);

                J = J ...
                    + 0.5*obj.Qx*ex^2 ...
                    + 0.5*obj.Qy*ey^2;
            end

            for j = 1:M
                h = tFine(j+1)-tFine(j);

                J = J + 0.5*h*( ...
                    obj.ChiV*W(1,j)^2 + ...
                    obj.ChiU*W(2,j)^2);
            end
        end

        function [g0,gW] = backwardSweep( ...
                obj,W,Z,M,Y,nMeas,tFine,uL,uM,uR)

            lambdaPlus = zeros(5,obj.MaxFineNodes);
            lambdaMinus = zeros(5,obj.MaxFineNodes);

            % No terminal cost.
            lambdaPlus(:,M+1) = zeros(5,1);

            % Final measurement jump.
            lambdaMinus(:,M+1) = ...
                lambdaPlus(:,M+1) + ...
                obj.measurementGradient( ...
                    Z(:,M+1),Y(:,nMeas));

            for j = M:-1:1
                h = -(tFine(j+1)-tFine(j));

                lamR = lambdaMinus(:,j+1);

                zR = Z(:,j+1);
                zL = Z(:,j);
                zM = 0.5*(zL+zR);

                k1 = obj.costateRHS(zR,lamR,uR(j));

                k2 = obj.costateRHS( ...
                    zM,lamR+0.5*h*k1,uM(j));

                k3 = obj.costateRHS( ...
                    zM,lamR+0.5*h*k2,uM(j));

                k4 = obj.costateRHS( ...
                    zL,lamR+h*k3,uL(j));

                lambdaPlus(:,j) = ...
                    lamR + (h/6)*(k1+2*k2+2*k3+k4);

                if mod(j-1,obj.NSub) == 0
                    kMeas = (j-1)/obj.NSub + 1;

                    if kMeas <= nMeas
                        lambdaMinus(:,j) = ...
                            lambdaPlus(:,j) + ...
                            obj.measurementGradient( ...
                                Z(:,j),Y(:,kMeas));
                    else
                        lambdaMinus(:,j) = lambdaPlus(:,j);
                    end
                else
                    lambdaMinus(:,j) = lambdaPlus(:,j);
                end
            end

            % No arrival cost:
            % dJ/dxi0 = lambda(t0^-)
            g0 = lambdaMinus(:,1);

            gW = zeros(2,obj.MaxM);

            for j = 1:M
                dtj = tFine(j+1)-tFine(j);

                lamAvg = 0.5*( ...
                    lambdaPlus(:,j) + ...
                    lambdaMinus(:,j+1));

                gW(1,j) = dtj*( ...
                    obj.ChiV*W(1,j) + lamAvg(4));

                gW(2,j) = dtj*( ...
                    obj.ChiU*W(2,j) + lamAvg(5));
            end
        end

        function g = measurementGradient(obj,z,yMeas)
            g = [ ...
                obj.Qx*(z(1)-yMeas(1)); ...
                obj.Qy*(z(2)-yMeas(2)); ...
                0; ...
                0; ...
                0];
        end

        function dz = relativeDynamics(~,z,w,vSelf,uSelf)
            x = z(1);
            y = z(2);
            theta = z(3);
            v1 = z(4);
            u1 = z(5);

            omegaV = w(1);
            omegaU = w(2);

            dz = [ ...
                v1*cos(theta) - vSelf + uSelf*y; ...
                v1*sin(theta)         - uSelf*x; ...
                u1-uSelf; ...
                omegaV; ...
                omegaU];
        end

        function dlambda = costateRHS(~,z,lambda,uSelf)
            theta = z(3);
            v1 = z(4);

            lx = lambda(1);
            ly = lambda(2);
            ltheta = lambda(3);

            dlambda = [ ...
                 uSelf*ly; ...
                -uSelf*lx; ...
                 v1*lx*sin(theta) - v1*ly*cos(theta); ...
                -lx*cos(theta) - ly*sin(theta); ...
                -ltheta];
        end

        function tFine = makeFineGrid(obj,nMeas)
            tFine = zeros(1,obj.MaxFineNodes);
            tFine(1) = obj.tBuf(1);

            idx = 1;

            for k = 1:nMeas-1
                ta = obj.tBuf(k);
                tb = obj.tBuf(k+1);
                h = (tb-ta)/obj.NSub;

                for s = 1:obj.NSub
                    idx = idx + 1;
                    tFine(idx) = ta + s*h;
                end
            end
        end

        function [vL,vM,vR,uL,uM,uR] = ...
                prepareKnownControls(obj,tFine,M)

            vL = zeros(1,obj.MaxM);
            vM = zeros(1,obj.MaxM);
            vR = zeros(1,obj.MaxM);

            uL = zeros(1,obj.MaxM);
            uM = zeros(1,obj.MaxM);
            uR = zeros(1,obj.MaxM);

            for j = 1:M
                tLeft = tFine(j);
                tRight = tFine(j+1);
                tMid = 0.5*(tLeft+tRight);

                [vL(j),uL(j)] = obj.interpolateKnownControl(tLeft);
                [vM(j),uM(j)] = obj.interpolateKnownControl(tMid);
                [vR(j),uR(j)] = obj.interpolateKnownControl(tRight);
            end
        end

        function recordControlSample(obj,t,vSelf,uSelf)
            if obj.nControl == 0
                obj.controlStart = 1;
                obj.nControl = 1;

                obj.controlTBuf(1) = t;
                obj.controlVBuf(1) = vSelf;
                obj.controlUBuf(1) = uSelf;
                return;
            end

            lastIdx = obj.controlPhysicalIndex(obj.nControl);
            lastT = obj.controlTBuf(lastIdx);

            tol = 1e-12*max(1.0,abs(t));

            if abs(t-lastT) <= tol
                obj.controlTBuf(lastIdx) = t;
                obj.controlVBuf(lastIdx) = vSelf;
                obj.controlUBuf(lastIdx) = uSelf;
                return;
            end

            % Ignore out-of-sequence samples.
            if t < lastT
                return;
            end

            if obj.nControl < obj.MaxControlSamples
                newIdx = obj.controlPhysicalIndex(obj.nControl+1);
                obj.nControl = obj.nControl + 1;
            else
                obj.controlStart = ...
                    mod(obj.controlStart,obj.MaxControlSamples) + 1;

                newIdx = obj.controlPhysicalIndex(obj.nControl);
            end

            obj.controlTBuf(newIdx) = t;
            obj.controlVBuf(newIdx) = vSelf;
            obj.controlUBuf(newIdx) = uSelf;
        end

        function [v,u] = interpolateKnownControl(obj,tq)
            if obj.nControl <= 0
                v = 0.0;
                u = 0.0;
                return;
            end

            idxFirst = obj.controlPhysicalIndex(1);
            tFirst = obj.controlTBuf(idxFirst);

            if obj.nControl == 1 || tq <= tFirst
                v = obj.controlVBuf(idxFirst);
                u = obj.controlUBuf(idxFirst);
                return;
            end

            idxLast = obj.controlPhysicalIndex(obj.nControl);
            tLast = obj.controlTBuf(idxLast);

            if tq >= tLast
                v = obj.controlVBuf(idxLast);
                u = obj.controlUBuf(idxLast);
                return;
            end

            for k = 1:obj.nControl-1
                idxA = obj.controlPhysicalIndex(k);
                idxB = obj.controlPhysicalIndex(k+1);

                ta = obj.controlTBuf(idxA);
                tb = obj.controlTBuf(idxB);

                if tq <= tb
                    if tb > ta
                        s = (tq-ta)/(tb-ta);
                    else
                        s = 0.0;
                    end

                    v = (1-s)*obj.controlVBuf(idxA) + ...
                         s *obj.controlVBuf(idxB);

                    u = (1-s)*obj.controlUBuf(idxA) + ...
                         s *obj.controlUBuf(idxB);
                    return;
                end
            end

            v = obj.controlVBuf(idxLast);
            u = obj.controlUBuf(idxLast);
        end

        function idx = controlPhysicalIndex(obj,k)
            idx = mod( ...
                obj.controlStart + k - 2, ...
                obj.MaxControlSamples) + 1;
        end

        function z = propagatePrediction( ...
                obj,z,t0,t1,vSelf0,uSelf0,vSelf1,uSelf1)

            totalDt = t1-t0;

            if totalDt <= 0
                return;
            end

            n = max(1,ceil(totalDt/obj.PredictionMaxStep));
            h = totalDt/n;

            w = zeros(2,1);

            for j = 1:n
                tau0 = (j-1)*h;
                tauM = tau0 + 0.5*h;
                tau1 = tau0 + h;

                s0 = tau0/totalDt;
                sM = tauM/totalDt;
                s1 = tau1/totalDt;

                v0 = (1-s0)*vSelf0 + s0*vSelf1;
                vm = (1-sM)*vSelf0 + sM*vSelf1;
                v1 = (1-s1)*vSelf0 + s1*vSelf1;

                u0 = (1-s0)*uSelf0 + s0*uSelf1;
                um = (1-sM)*uSelf0 + sM*uSelf1;
                u1 = (1-s1)*uSelf0 + s1*uSelf1;

                k1 = obj.relativeDynamics(z,w,v0,u0);

                k2 = obj.relativeDynamics( ...
                    z+0.5*h*k1,w,vm,um);

                k3 = obj.relativeDynamics( ...
                    z+0.5*h*k2,w,vm,um);

                k4 = obj.relativeDynamics( ...
                    z+h*k3,w,v1,u1);

                z = z + ...
                    (h/6)*(k1+2*k2+2*k3+k4);
            end

            z(3) = obj.wrapPi(z(3));
        end

        function a = wrapPi(~,a)
            a = atan2(sin(a),cos(a));
        end
    end
end
