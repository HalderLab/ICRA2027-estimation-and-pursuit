clc
clear
close all

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% ========================================================================
%
% Open-loop simulation benchmark:
%   true v1(t) = 0.13 + 0.03*sin(0.30*t) m/s
%   true u1(t) = -0.164 + 0.03*sin(0.10*t) rad/s
%
% All three estimators solve the same unknown-state problem:
%   PF  -> [rho, alpha1, alpha2, v1, u1]
%   EKF -> [rho, alpha1, alpha2, v1, u1]
%   IME -> IMEBlock, which estimates [x,y,theta,v1,u1]
%
% PF/EKF model v1 and u1 as random walks:
%   v1_dot = 0 plus process noise
%   u1_dot = 0 plus process noise
%
% IME uses the Simulink IMEBlock formulation:
%   v1_dot = a1,  u1_dot = omega1
% and optimizes a1 and omega1 with its native chi_v and chi_u penalties.
%
% PF and EKF use the same process-noise standard deviations numerically.
% The PF samples those standard deviations directly; EKF Q contains their
% squared values.
%
% The script:
%   1. generates ONE prescribed varying-speed physical truth trajectory
%   2. safety-checks robot separation
%   3. runs Monte Carlo estimator trials with varying estimator initial guesses
%   4. plots one dashboard in the same style as the physical benchmark
%   5. saves singleVaryingResults

%% 1. User configuration

cfg.dt = 0.10;
cfg.nPlantSub = 5;
cfg.duration = 40;
cfg.numTrials = 100; 
cfg.baseSeed = 2026;
cfg.percentileLow = 10;
cfg.percentileHigh = 90;

% Pose error:
%   e_g = || I3 - g_true^(-1) g_est ||
% Use 'fro' for the Frobenius norm. If a different matrix norm is requested,
% change only this setting.
cfg.poseErrorNorm = 'fro';

% Initial global state q = [x1;y1;theta1;x2;y2;theta2].
cfg.q0 = [-0.824; -0.200; pi/2; -0.450; -0.057; pi/2];

% Prescribed single varying-speed condition.
% IMPORTANT: the requested law is sin(0.30*t), so 0.30 is used as
% angular frequency in rad/s. It is NOT interpreted as 0.30 Hz.
cfg.v1Base = 0.13 ;
cfg.v1Amplitude = 0.03;
cfg.v1Omega = 0.30;
cfg.u1Base = -0.164;
cfg.u1Amplitude = 0.03;
cfg.u1Omega = 0.10;
cfg.v2 = 0.070;
cfg.u2Const = -0.155;

% Physical safety check.
cfg.minimumAllowedSeparation = 0.25;

% Synthetic LiDAR noise.
cfg.sigmaRho = 0.025;
cfg.sigmaAlpha2 = 0.02;
cfg.rhoMeasurementFloor = 1e-3;
cfg.rhoModelFloor = 0.03;

% Hidden-state prior information.
cfg.alpha1PriorRange = [-pi pi];
cfg.v1PriorRange = [0.00 0.22];
cfg.u1PriorRange = [-0.50 0.50];

% PF native settings.
cfg.pf.numParticles = 35000;
cfg.pf.initialRhoStd = 0.05;
cfg.pf.initialAlpha2Std = 0.10;
cfg.pf.resampleFraction = 0.50;
cfg.pf.nominalDt = cfg.dt;

% EKF native initialization.
cfg.ekf.P0 = diag([0.05^2 0.80^2 0.10^2 0.10^2 0.40^2]);
cfg.ekf.nominalDt = cfg.dt;
cfg.ekf.H = [1 0 0 0 0;0 0 1 0 0];
cfg.ekf.R = diag([cfg.sigmaRho^2 cfg.sigmaAlpha2^2]);

% Matched PF/EKF process noise, preserving the uploaded benchmark's direct
% process-noise structure on rho, alpha1, alpha2, and v1.
cfg.processStdRho = 0.05; 
cfg.processStdAlpha1 = 0.087; 
cfg.processStdAlpha2 = 0.087; 
cfg.processStdV1 = 0.002;
cfg.processStdU1 = 0.002; 
cfg.commonProcessStd = [cfg.processStdRho;cfg.processStdAlpha1;cfg.processStdAlpha2;cfg.processStdV1;cfg.processStdU1];
cfg.pf.processStd = cfg.commonProcessStd;
cfg.ekf.Q0 = diag(cfg.commonProcessStd.^2);
cfg.pf.sigmaRho = cfg.sigmaRho;
cfg.pf.sigmaAlpha2 = cfg.sigmaAlpha2;

% Parallel Computing Toolbox.
% Leave empty to let MATLAB choose the default worker count.
cfg.useParallel = true;
cfg.numWorkers = [];

% Save configuration.
cfg.saveResults = true;
cfg.resultsFolder = 'open_loop_results';
cfg.resultsPrefix = 'open_loop_results';

% Plot colors from the physical benchmark.
cfg.colors.truth = [0.05 0.05 0.05];
cfg.colors.ekf = [0.00 0.4470 0.7410];
cfg.colors.pf = [0.8500 0.3250 0.0980];
cfg.colors.mhe = [0.4660 0.6740 0.1880];
cfg.colors.leader = [0.4940 0.1840 0.5560];
cfg.colors.follower = [0.6350 0.0780 0.1840];

%% 2. Verify IMEBlock

if exist('IMEBlock','class') ~= 8
    error('IMEBlock.m must be on the MATLAB path or in the same folder.')
end

imeMethods = methods('IMEBlock');
if ~any(strcmp(imeMethods,'setInitialGuess'))
    error(['IMEBlock.m must contain setInitialGuess(thetaGuess,v1Guess,u1Guess) ' ...
           'for the Monte Carlo initialization study.']);
end

%% 2B. Start parallel pool

if cfg.useParallel
    pool = gcp('nocreate');

    if isempty(pool)
        if isempty(cfg.numWorkers)
            pool = parpool;
        else
            pool = parpool(cfg.numWorkers);
        end
    end

    fprintf('Parallel pool active with %d workers.\n',pool.NumWorkers)
else
    fprintf('Parallel execution disabled. Using ordinary for-loops.\n')
end

%% 3. Generate the one requested varying-speed physical truth trajectory

truth = generateTruthSingleVarying(cfg);

dx = truth.q(1,:)-truth.q(4,:);
dy = truth.q(2,:)-truth.q(5,:);
separation = hypot(dx,dy);
minimumSeparation = min(separation);

fprintf('\n============================================================\n')
fprintf('Single varying-v1/u1 open-loop benchmark: PF vs EKF vs IMEBlock\n')
fprintf('============================================================\n')
fprintf('v1(t) = %.3f + %.3f*sin(%.3f*t) m/s\n',cfg.v1Base,cfg.v1Amplitude,cfg.v1Omega)
fprintf('u1(t) = %.3f + %.3f*sin(%.3f*t) rad/s\n',cfg.u1Base,cfg.u1Amplitude,cfg.u1Omega)
fprintf('v2 = %.4f m/s\n',cfg.v2)
fprintf('u2 = %.4f rad/s\n',cfg.u2Const)
fprintf('Duration = %.2f s\n',cfg.duration)
fprintf('Minimum prescribed robot separation = %.3f m\n',minimumSeparation)

if minimumSeparation < cfg.minimumAllowedSeparation
    error('Requested varying-speed trajectory violates cfg.minimumAllowedSeparation.')
end

%% 4. Monte Carlo hidden-state point guesses for EKF and IME

trialGuess = repmat(struct('alpha1',NaN,'v1',NaN,'u1',NaN),cfg.numTrials,1);

for trial = 1:cfg.numTrials
    stream = RandStream('mt19937ar','Seed',cfg.baseSeed+100000+trial);
    trialGuess(trial).alpha1 = cfg.alpha1PriorRange(1) + diff(cfg.alpha1PriorRange)*rand(stream,1);
    trialGuess(trial).v1 = cfg.v1PriorRange(1) + diff(cfg.v1PriorRange)*rand(stream,1);
    trialGuess(trial).u1 = cfg.u1PriorRange(1) + diff(cfg.u1PriorRange)*rand(stream,1);
end

%% 5. Run Monte Carlo trials

K = numel(truth.t);

% Use plain sliced matrices inside PARFOR. After the parallel loop, rebuild
% the same errorTrials structure used by the rest of this file.
errorXYPF = nan(K,cfg.numTrials);
errorXYEKF = nan(K,cfg.numTrials);
errorXYMHE = nan(K,cfg.numTrials);

errorThetaPF = nan(K,cfg.numTrials);
errorThetaEKF = nan(K,cfg.numTrials);
errorThetaMHE = nan(K,cfg.numTrials);

errorPosePF = nan(K,cfg.numTrials);
errorPoseEKF = nan(K,cfg.numTrials);
errorPoseMHE = nan(K,cfg.numTrials);

errorV1PF = nan(K,cfg.numTrials);
errorV1EKF = nan(K,cfg.numTrials);
errorV1MHE = nan(K,cfg.numTrials);

errorU1PF = nan(K,cfg.numTrials);
errorU1EKF = nan(K,cfg.numTrials);
errorU1MHE = nan(K,cfg.numTrials);

scoreStartTime = nan(cfg.numTrials,1);
pfResamplingEvents = nan(cfg.numTrials,1);

fprintf('\nTrials per condition: %d\n',cfg.numTrials)
fprintf('PF particles: %d\n',cfg.pf.numParticles)
fprintf('PF hidden-state initialization: broad particles over admissible ranges\n')
fprintf('EKF/IME hidden-state initialization: one shared random point guess per trial\n')
fprintf('Matched PF/EKF process stds at dt = %.3f s:\n',cfg.dt)
fprintf('  rho    = %.6g m\n',cfg.processStdRho)
fprintf('  alpha1 = %.6g rad = %.3f deg\n',cfg.processStdAlpha1,rad2deg(cfg.processStdAlpha1))
fprintf('  alpha2 = %.6g rad = %.3f deg\n',cfg.processStdAlpha2,rad2deg(cfg.processStdAlpha2))
fprintf('  v1     = %.6g m/s\n',cfg.processStdV1)
fprintf('  u1     = %.6g rad/s\n\n',cfg.processStdU1)

if cfg.useParallel
    parallelProgress('single varying',cfg.numTrials,true);
    progressQueue = parallel.pool.DataQueue;
    afterEach(progressQueue,@(~) parallelProgress('single varying',cfg.numTrials,false));

    parfor trial = 1:cfg.numTrials
        measurementSeed = cfg.baseSeed+200000+trial;
        pfSeed = cfg.baseSeed+300000+trial;

        meas = generateMeasurements(truth,cfg,measurementSeed);
        run = runThreeEstimators(truth,meas,trialGuess(trial),cfg,pfSeed,0.0);

        errorXYPF(:,trial) = run.error.xy.PF;
        errorXYEKF(:,trial) = run.error.xy.EKF;
        errorXYMHE(:,trial) = run.error.xy.MHE;

        errorThetaPF(:,trial) = run.error.theta.PF;
        errorThetaEKF(:,trial) = run.error.theta.EKF;
        errorThetaMHE(:,trial) = run.error.theta.MHE;

        errorPosePF(:,trial) = run.error.pose.PF;
        errorPoseEKF(:,trial) = run.error.pose.EKF;
        errorPoseMHE(:,trial) = run.error.pose.MHE;

        errorV1PF(:,trial) = run.error.v1.PF;
        errorV1EKF(:,trial) = run.error.v1.EKF;
        errorV1MHE(:,trial) = run.error.v1.MHE;

        errorU1PF(:,trial) = run.error.u1.PF;
        errorU1EKF(:,trial) = run.error.u1.EKF;
        errorU1MHE(:,trial) = run.error.u1.MHE;

        scoreStartTime(trial) = run.commonScoreStartTime;
        pfResamplingEvents(trial) = run.pfResamplingEvents;

        send(progressQueue,trial)
    end
else
    for trial = 1:cfg.numTrials
        measurementSeed = cfg.baseSeed+200000+trial;
        pfSeed = cfg.baseSeed+300000+trial;

        meas = generateMeasurements(truth,cfg,measurementSeed);
        run = runThreeEstimators(truth,meas,trialGuess(trial),cfg,pfSeed,0.0);

        errorXYPF(:,trial) = run.error.xy.PF;
        errorXYEKF(:,trial) = run.error.xy.EKF;
        errorXYMHE(:,trial) = run.error.xy.MHE;

        errorThetaPF(:,trial) = run.error.theta.PF;
        errorThetaEKF(:,trial) = run.error.theta.EKF;
        errorThetaMHE(:,trial) = run.error.theta.MHE;

        errorPosePF(:,trial) = run.error.pose.PF;
        errorPoseEKF(:,trial) = run.error.pose.EKF;
        errorPoseMHE(:,trial) = run.error.pose.MHE;

        errorV1PF(:,trial) = run.error.v1.PF;
        errorV1EKF(:,trial) = run.error.v1.EKF;
        errorV1MHE(:,trial) = run.error.v1.MHE;

        errorU1PF(:,trial) = run.error.u1.PF;
        errorU1EKF(:,trial) = run.error.u1.EKF;
        errorU1MHE(:,trial) = run.error.u1.MHE;

        scoreStartTime(trial) = run.commonScoreStartTime;
        pfResamplingEvents(trial) = run.pfResamplingEvents;

        fprintf('  single varying trial %d / %d complete\n',trial,cfg.numTrials)
    end
end

errorTrials.xy.PF = errorXYPF;
errorTrials.xy.EKF = errorXYEKF;
errorTrials.xy.MHE = errorXYMHE;

errorTrials.theta.PF = errorThetaPF;
errorTrials.theta.EKF = errorThetaEKF;
errorTrials.theta.MHE = errorThetaMHE;

errorTrials.pose.PF = errorPosePF;
errorTrials.pose.EKF = errorPoseEKF;
errorTrials.pose.MHE = errorPoseMHE;

errorTrials.v1.PF = errorV1PF;
errorTrials.v1.EKF = errorV1EKF;
errorTrials.v1.MHE = errorV1MHE;

errorTrials.u1.PF = errorU1PF;
errorTrials.u1.EKF = errorU1EKF;
errorTrials.u1.MHE = errorU1MHE;

%% 6. Package results

singleVaryingResults = struct();
singleVaryingResults.cfg = cfg;
singleVaryingResults.truth = truth;
singleVaryingResults.minimumSeparation = minimumSeparation;
singleVaryingResults.trialGuess = trialGuess;
singleVaryingResults.errorTrials = errorTrials;
singleVaryingResults.scoreStartTime = scoreStartTime;
singleVaryingResults.pfResamplingEvents = pfResamplingEvents;
singleVaryingResults.savedAt = datetime('now');

%% 7. Plotting

% Plotting is intentionally omitted here.
% Use plot_open_loop_simulated.m to load and plot the saved results.

%% 8. Save complete result structure

if cfg.saveResults
    if ~isfolder(cfg.resultsFolder)
        mkdir(cfg.resultsFolder)
    end

    saveStamp = datestr(now,'yyyymmdd_HHMMSS');
    saveFile = fullfile(cfg.resultsFolder,sprintf('%s_%s.mat',cfg.resultsPrefix,saveStamp));
    save(saveFile,'singleVaryingResults','-v7.3');
    fprintf('\nSaved single varying benchmark results:\n  %s\n',saveFile)
end

fprintf('\nSingle varying-v1/u1 simulation complete. No figures were generated.\n')

%% =========================================================================
% LOCAL FUNCTIONS

function truth = generateTruthSingleVarying(cfg)

t = 0:cfg.dt:cfg.duration;
K = numel(t);
q = nan(6,K);
q(:,1) = cfg.q0;

v1Fun = @(tt) cfg.v1Base + cfg.v1Amplitude*sin(cfg.v1Omega*tt);
u1Fun = @(tt) cfg.u1Base + cfg.u1Amplitude*sin(cfg.u1Omega*tt);

for k = 1:K-1
    q(:,k+1) = propagatePrescribedPair(q(:,k),t(k),cfg.dt,cfg.nPlantSub,v1Fun,u1Fun,cfg.v2,cfg.u2Const);
end

rho = nan(1,K);
alpha1 = nan(1,K);
alpha2 = nan(1,K);
xRel = nan(1,K);
yRel = nan(1,K);
thetaRel = nan(1,K);

for k = 1:K
    [rho(k),alpha1(k),alpha2(k),xRel(k),yRel(k),thetaRel(k)] = relativeTruth(q(:,k));
end

truth.t = t;
truth.q = q;
truth.rho = rho;
truth.alpha1 = alpha1;
truth.alpha2 = alpha2;
truth.x = xRel;
truth.y = yRel;
truth.theta = thetaRel;
truth.v1 = cfg.v1Base + cfg.v1Amplitude*sin(cfg.v1Omega*t);
truth.u1 = cfg.u1Base + cfg.u1Amplitude*sin(cfg.u1Omega*t);
truth.v2 = cfg.v2*ones(1,K);
truth.u2 = cfg.u2Const*ones(1,K);

end

function meas = generateMeasurements(truth,cfg,seed)

stream = RandStream('mt19937ar','Seed',seed);
noiseRho = cfg.sigmaRho*randn(stream,1,numel(truth.t));
noiseAlpha2 = cfg.sigmaAlpha2*randn(stream,1,numel(truth.t));

meas.rho = max(truth.rho + noiseRho,cfg.rhoMeasurementFloor);
meas.alpha2 = wrapPi(truth.alpha2 + noiseAlpha2);
meas.noiseRho = noiseRho;
meas.noiseAlpha2 = noiseAlpha2;

end

function run = runThreeEstimators(truth,meas,pointGuess,cfg,pfSeed,requestedScoreStart)

K = numel(truth.t);

% -------------------------------------------------------------------------
% PF: broad-support prior over both hidden motion states v1 and u1

rng(pfSeed,'twister')

N = cfg.pf.numParticles;
particles = zeros(N,5);
particles(:,1) = max(meas.rho(1) + cfg.pf.initialRhoStd*randn(N,1),cfg.rhoModelFloor);
particles(:,2) = cfg.alpha1PriorRange(1) + diff(cfg.alpha1PriorRange)*rand(N,1);
particles(:,2) = wrapPi(particles(:,2));
particles(:,3) = wrapPi(meas.alpha2(1) + cfg.pf.initialAlpha2Std*randn(N,1));
particles(:,4) = cfg.v1PriorRange(1) + diff(cfg.v1PriorRange)*rand(N,1);
particles(:,5) = cfg.u1PriorRange(1) + diff(cfg.u1PriorRange)*rand(N,1);

weights = ones(N,1)/N;
xHatPF = nan(K,5);
xHatPF(1,:) = weightedParticleMean5(particles,weights);
pfResamplingEvents = 0;

% -------------------------------------------------------------------------
% EKF: same 5-state estimation problem

xHatEKF = nan(K,5);
xHatEKF(1,:) = [max(meas.rho(1),cfg.rhoModelFloor) wrapPi(pointGuess.alpha1) wrapPi(meas.alpha2(1)) pointGuess.v1 pointGuess.u1];
P = cfg.ekf.P0;
I5 = eye(5);

% -------------------------------------------------------------------------
% IME: use the same IMEBlock implementation used by the Simulink model.
% The Monte Carlo alpha1 guess is converted to the block's internal theta
% convention using theta = alpha2 + pi - alpha1, exactly as before.

thetaInitialGuess = wrapPi(meas.alpha2(1) + pi - pointGuess.alpha1);
ime = IMEBlock();
ime.setInitialGuess(thetaInitialGuess,pointGuess.v1,pointGuess.u1);

xHatMHE = nan(K,5);   % Keep the legacy MHE field name for plot/result compatibility.
readyMHE = false(K,1);
horizonSamples = zeros(K,1);
horizonDuration = zeros(K,1);

for k = 1:K

    yIME = [meas.rho(k);meas.alpha2(k)];
    eta2IME = [cfg.v2;cfg.u2Const];
    newMeasurement = 1;

    [zHatIME,eta1HatIME,readyIME,~,~] = ...
        step(ime,truth.t(k),yIME,eta2IME,newMeasurement);

    readyMHE(k) = logical(readyIME);

    % IMEBlock intentionally keeps its Simulink output interface compact,
    % so reproduce the old growMHE horizon diagnostics here.  This benchmark
    % supplies one LiDAR sample every cfg.dt and the block uses Nh = 30.
    horizonSamples(k) = min(k,31);
    leftIndex = max(1,k-30);
    horizonDuration(k) = truth.t(k)-truth.t(leftIndex);

    if readyMHE(k)
        xIME = zHatIME(1);
        yIMEHat = zHatIME(2);
        thetaIME = wrapPi(zHatIME(3));

        rhoIME = hypot(xIME,yIMEHat);
        alpha2IME = wrapPi(atan2(yIMEHat,xIME));
        alpha1IME = wrapPi(alpha2IME + pi - thetaIME);

        xHatMHE(k,:) = [rhoIME alpha1IME alpha2IME eta1HatIME(1) eta1HatIME(2)];
    end

    if k == 1
        continue
    end

    dt = truth.t(k)-truth.t(k-1);

    % ---------------------------------------------------------------------
    % PF prediction
    %
    % v1 and u1 each follow a discrete random walk:
    %   v1(k+1) = v1(k) + noise
    %   u1(k+1) = u1(k) + noise

    scale = sqrt(max(dt/cfg.pf.nominalDt,1e-12));
    ps = cfg.pf.processStd*scale;

    rho = max(particles(:,1),cfg.rhoModelFloor);
    a1 = particles(:,2);
    a2 = particles(:,3);
    v1p = particles(:,4);
    u1p = particles(:,5);

    shared = v1p.*sin(a1) + cfg.v2.*sin(a2);
    rhoDot = -v1p.*cos(a1) - cfg.v2.*cos(a2);
    a1Dot = -u1p + shared./rho;
    a2Dot = -cfg.u2Const + shared./rho;

    particles(:,1) = max(rho + dt*rhoDot + ps(1)*randn(N,1),cfg.rhoModelFloor);
    particles(:,2) = wrapPi(a1 + dt*a1Dot + ps(2)*randn(N,1));
    particles(:,3) = wrapPi(a2 + dt*a2Dot + ps(3)*randn(N,1));
    particles(:,4) = v1p + ps(4)*randn(N,1);
    particles(:,5) = u1p + ps(5)*randn(N,1);

    % Resolve the equivalent (alpha1 + pi, -v1) branch instead of
    % clamping negative speed to zero.  This matches the IME output
    % convention while preserving the represented Cartesian motion.
    negV1 = particles(:,4) < 0;
    particles(negV1,4) = -particles(negV1,4);
    particles(negV1,2) = wrapPi(particles(negV1,2) + pi);

    er = meas.rho(k)-particles(:,1);
    ea = wrapPi(meas.alpha2(k)-particles(:,3));
    logLike = -0.5*(er/cfg.pf.sigmaRho).^2 - 0.5*(ea/cfg.pf.sigmaAlpha2).^2;
    logLike = logLike-max(logLike);
    like = exp(logLike);

    weights = weights.*like;
    sw = sum(weights);

    if ~isfinite(sw) || sw <= realmin
        weights = ones(N,1)/N;
    else
        weights = weights/sw;
    end

    xHatPF(k,:) = weightedParticleMean5(particles,weights);

    Neff = 1/sum(weights.^2);

    if Neff < cfg.pf.resampleFraction*N
        idx = systematicResampleIndices(weights);
        particles = particles(idx,:);
        weights = ones(N,1)/N;
        pfResamplingEvents = pfResamplingEvents+1;
    end

    % ---------------------------------------------------------------------
    % EKF prediction + correction
    %
    % The deterministic model has v1_dot = 0 and u1_dot = 0. Their Q
    % entries create the corresponding random-walk uncertainty.

    x = xHatEKF(k-1,:)';
    [f,Fc] = shapeDynamicsJacobian5(x,cfg.v2,cfg.u2Const,cfg.rhoModelFloor);

    xPred = x + dt*f;
    xPred(1) = max(xPred(1),cfg.rhoModelFloor);
    xPred(2) = wrapPi(xPred(2));
    xPred(3) = wrapPi(xPred(3));

    % Use the same positive-speed branch convention as the IME.  If the
    % predicted mean lands on v1 < 0, the equivalent representation is
    % obtained by v1 -> -v1 and alpha1 -> alpha1 + pi.
    predBranchFlip = xPred(4) < 0;
    if predBranchFlip
        xPred(4) = -xPred(4);
        xPred(2) = wrapPi(xPred(2) + pi);
    end

    A = I5 + dt*Fc;
    qScale = max(dt/cfg.ekf.nominalDt,1e-12);
    Q = cfg.ekf.Q0*qScale;

    PPred = A*P*A' + Q;

    % Transform the covariance under v1 -> -v1 when the branch is flipped.
    if predBranchFlip
        Tbranch = diag([1 1 1 -1 1]);
        PPred = Tbranch*PPred*Tbranch';
    end

    PPred = 0.5*(PPred+PPred');

    zPred = [xPred(1);xPred(3)];
    innov = [meas.rho(k)-zPred(1);wrapPi(meas.alpha2(k)-zPred(2))];

    S = cfg.ekf.H*PPred*cfg.ekf.H' + cfg.ekf.R;
    S = 0.5*(S+S') + 1e-12*eye(2);
    Kgain = (PPred*cfg.ekf.H')/S;

    xNew = xPred + Kgain*innov;
    xNew(1) = max(xNew(1),cfg.rhoModelFloor);
    xNew(2) = wrapPi(xNew(2));
    xNew(3) = wrapPi(xNew(3));

    P = (I5-Kgain*cfg.ekf.H)*PPred*(I5-Kgain*cfg.ekf.H)' + Kgain*cfg.ekf.R*Kgain';

    % Resolve the equivalent negative-v1 branch after the measurement
    % correction as well, and transform the covariance consistently.
    if xNew(4) < 0
        xNew(4) = -xNew(4);
        xNew(2) = wrapPi(xNew(2) + pi);
        Tbranch = diag([1 1 1 -1 1]);
        P = Tbranch*P*Tbranch';
    end

    P = 0.5*(P+P');

    xHatEKF(k,:) = xNew';
end

firstReadyIndex = find(readyMHE,1,'first');

if isempty(firstReadyIndex)
    error('IMEBlock never became ready in this run.')
end

commonScoreStartTime = max(requestedScoreStart,truth.t(firstReadyIndex));
scoreMask = truth.t >= commonScoreStartTime;

truthCommon = shapeToCommon(truth.rho,truth.alpha1,truth.alpha2,truth.v1,truth.u1);
pfCommon = shapeToCommon(xHatPF(:,1)',xHatPF(:,2)',xHatPF(:,3)',xHatPF(:,4)',xHatPF(:,5)');
ekfCommon = shapeToCommon(xHatEKF(:,1)',xHatEKF(:,2)',xHatEKF(:,3)',xHatEKF(:,4)',xHatEKF(:,5)');
mheCommon = shapeToCommon(xHatMHE(:,1)',xHatMHE(:,2)',xHatMHE(:,3)',xHatMHE(:,4)',xHatMHE(:,5)');

errorXY.PF = sqrt((pfCommon(1,:)-truthCommon(1,:)).^2 + (pfCommon(2,:)-truthCommon(2,:)).^2)';
errorXY.EKF = sqrt((ekfCommon(1,:)-truthCommon(1,:)).^2 + (ekfCommon(2,:)-truthCommon(2,:)).^2)';
errorXY.MHE = sqrt((mheCommon(1,:)-truthCommon(1,:)).^2 + (mheCommon(2,:)-truthCommon(2,:)).^2)';

errorTheta.PF = abs(wrapPi(pfCommon(3,:)-truthCommon(3,:)))';
errorTheta.EKF = abs(wrapPi(ekfCommon(3,:)-truthCommon(3,:)))';
errorTheta.MHE = abs(wrapPi(mheCommon(3,:)-truthCommon(3,:)))';

errorPose.PF = poseTransformErrorSeries(truthCommon,pfCommon,cfg.poseErrorNorm);
errorPose.EKF = poseTransformErrorSeries(truthCommon,ekfCommon,cfg.poseErrorNorm);
errorPose.MHE = poseTransformErrorSeries(truthCommon,mheCommon,cfg.poseErrorNorm);

errorV1.PF  = (abs(pfCommon(4,:)  - truthCommon(4,:)) ./ abs(truthCommon(4,:)))';
errorV1.EKF = (abs(ekfCommon(4,:) - truthCommon(4,:)) ./ abs(truthCommon(4,:)))';
errorV1.MHE = (abs(mheCommon(4,:) - truthCommon(4,:)) ./ abs(truthCommon(4,:)))';

errorU1.PF  = (abs(pfCommon(5,:)  - truthCommon(5,:)) ./ abs(truthCommon(5,:)))';
errorU1.EKF = (abs(ekfCommon(5,:) - truthCommon(5,:)) ./ abs(truthCommon(5,:)))';
errorU1.MHE = (abs(mheCommon(5,:) - truthCommon(5,:)) ./ abs(truthCommon(5,:)))'; 

names = {'PF','EKF','MHE'};

for ii = 1:numel(names)
    name = names{ii};
    errorXY.(name)(~scoreMask) = NaN;
    errorTheta.(name)(~scoreMask) = NaN;
    errorPose.(name)(~scoreMask) = NaN;
    errorV1.(name)(~scoreMask) = NaN;
    errorU1.(name)(~scoreMask) = NaN;
end

run = struct();
run.commonScoreStartTime = commonScoreStartTime;
run.readyMHE = readyMHE;
run.horizonSamples = horizonSamples;
run.horizonDuration = horizonDuration;
run.pfResamplingEvents = pfResamplingEvents;
run.error.xy = errorXY;
run.error.theta = errorTheta;
run.error.pose = errorPose;
run.error.v1 = errorV1;
run.error.u1 = errorU1;
run.estimate.PF = xHatPF;
run.estimate.EKF = xHatEKF;
run.estimate.MHE = xHatMHE;  % IMEBlock estimates; legacy field name retained

end

function [f,Fc] = shapeDynamicsJacobian5(x,v2,u2,rhoMin)

rho = max(x(1),rhoMin);
a1 = x(2);
a2 = x(3);
v1 = x(4);
u1 = x(5);

shared = v1*sin(a1) + v2*sin(a2);

f = [-v1*cos(a1)-v2*cos(a2);-u1+shared/rho;-u2+shared/rho;0;0];

Fc = zeros(5,5);
Fc(1,2) = v1*sin(a1);
Fc(1,3) = v2*sin(a2);
Fc(1,4) = -cos(a1);

Fc(2,1) = -shared/(rho^2);
Fc(2,2) = v1*cos(a1)/rho;
Fc(2,3) = v2*cos(a2)/rho;
Fc(2,4) = sin(a1)/rho;
Fc(2,5) = -1;

Fc(3,1) = Fc(2,1);
Fc(3,2) = Fc(2,2);
Fc(3,3) = Fc(2,3);
Fc(3,4) = Fc(2,4);

end

function common = shapeToCommon(rho,alpha1,alpha2,v1,u1)

x = rho.*cos(alpha2);
y = rho.*sin(alpha2);
theta = wrapPi(alpha2 + pi - alpha1);
common = [x;y;theta;v1;u1];

end

function errorSeries = poseTransformErrorSeries(truthCommon,estimateCommon,normType)

K = size(truthCommon,2);
errorSeries = nan(K,1);
I3 = eye(3);

for k = 1:K
    truePose = truthCommon(1:3,k);
    estPose = estimateCommon(1:3,k);

    if any(~isfinite(truePose)) || any(~isfinite(estPose))
        continue
    end

    xTrue = truePose(1);
    yTrue = truePose(2);
    thetaTrue = truePose(3);

    xEst = estPose(1);
    yEst = estPose(2);
    thetaEst = estPose(3);

    gTrue = [cos(thetaTrue) -sin(thetaTrue) xTrue; sin(thetaTrue) cos(thetaTrue) yTrue; 0 0 1];
    gEst = [cos(thetaEst) -sin(thetaEst) xEst; sin(thetaEst) cos(thetaEst) yEst; 0 0 1];

    relativeErrorTransform = gTrue \ gEst;
    errorSeries(k) = norm(I3-relativeErrorTransform,normType);
end

end

function x = weightedParticleMean5(particles,weights)

rho = sum(weights.*particles(:,1));
a1 = circularMean(particles(:,2),weights);
a2 = circularMean(particles(:,3),weights);
v1 = sum(weights.*particles(:,4));
u1 = sum(weights.*particles(:,5));
x = [rho a1 a2 v1 u1];

end

function a = circularMean(theta,w)

a = atan2(sum(w.*sin(theta)),sum(w.*cos(theta)));

end

function idx = systematicResampleIndices(weights)

N = numel(weights);
positions = ((0:N-1)' + rand)/N;
c = cumsum(weights);
c(end) = 1;
idx = zeros(N,1);
i = 1;
j = 1;

while i <= N
    if positions(i) <= c(j)
        idx(i) = j;
        i = i+1;
    else
        j = j+1;
    end
end

end

function qNext = propagatePrescribedPair(q0,t0,dt,nSub,v1Fun,u1Fun,v2,u2)

q = q0;
h = dt/nSub;

for jj = 1:nSub
    ta = t0 + (jj-1)*h;
    tm = ta + 0.5*h;
    tb = ta + h;

    k1 = pairRhs(q,v1Fun(ta),u1Fun(ta),v2,u2);
    k2 = pairRhs(q+0.5*h*k1,v1Fun(tm),u1Fun(tm),v2,u2);
    k3 = pairRhs(q+0.5*h*k2,v1Fun(tm),u1Fun(tm),v2,u2);
    k4 = pairRhs(q+h*k3,v1Fun(tb),u1Fun(tb),v2,u2);

    q = q + (h/6)*(k1+2*k2+2*k3+k4);
end

q(3) = wrapPi(q(3));
q(6) = wrapPi(q(6));
qNext = q;

end

function dq = pairRhs(q,v1,u1,v2,u2)

dq = [v1*cos(q(3));v1*sin(q(3));u1;v2*cos(q(6));v2*sin(q(6));u2];

end

function [rho,alpha1,alpha2,xRel,yRel,thetaRel] = relativeTruth(q)

p1 = q(1:2);
theta1 = q(3);
p2 = q(4:5);
theta2 = q(6);

r21Global = p1-p2;
R2T = [cos(theta2) sin(theta2);-sin(theta2) cos(theta2)];
r21Body2 = R2T*r21Global;

xRel = r21Body2(1);
yRel = r21Body2(2);
rho = hypot(xRel,yRel);
alpha2 = wrapPi(atan2(yRel,xRel));

r12Global = -r21Global;
los12 = atan2(r12Global(2),r12Global(1));
alpha1 = wrapPi(los12-theta1);

thetaRel = wrapPi(theta1-theta2);

end

function [med,low,high] = medianPercentile(data,pLow,pHigh)

nRows = size(data,1);
med = nan(nRows,1);
low = nan(nRows,1);
high = nan(nRows,1);

for rr = 1:nRows
    vals = data(rr,:);
    vals = vals(isfinite(vals));

    if isempty(vals)
        continue
    end

    med(rr) = median(vals);
    low(rr) = percentile1D(vals,pLow);
    high(rr) = percentile1D(vals,pHigh);
end

end

function q = percentile1D(vals,p)

vals = sort(vals(:));
n = numel(vals);

if n == 1
    q = vals(1);
    return
end

position = 1 + (n-1)*(p/100);
lo = floor(position);
hi = ceil(position);

if lo == hi
    q = vals(lo);
else
    fraction = position-lo;
    q = vals(lo) + fraction*(vals(hi)-vals(lo));
end

end

function value = rmseFinite(errorVector)

vals = errorVector(isfinite(errorVector));

if isempty(vals)
    value = NaN;
else
    value = sqrt(mean(vals.^2));
end

end

function plotEstimatorEnvelopes(ax,x,dataStruct,cfg)

names = {'PF','EKF','MHE'};
colors = [cfg.colors.pf;cfg.colors.ekf;cfg.colors.mhe];

for ii = 1:numel(names)
    name = names{ii};
    [center,low,high] = medianPercentile(dataStruct.(name),cfg.percentileLow,cfg.percentileHigh);

    xv = x(:);
    center = center(:);
    low = low(:);
    high = high(:);

    valid = isfinite(xv) & isfinite(center) & isfinite(low) & isfinite(high);

    if ~any(valid)
        continue
    end

    xv = xv(valid);
    center = center(valid);
    low = low(valid);
    high = high(valid);
    color = colors(ii,:);

    fill(ax,[xv;flipud(xv)],[low;flipud(high)],color,'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off')
    plot(ax,xv,center,'-','Color',color,'LineWidth',2.0,'DisplayName',name)
end

end

function parallelProgress(label,total,reset)

persistent count currentLabel

if reset || isempty(count) || isempty(currentLabel) || ~strcmp(currentLabel,label)
    count = 0;
    currentLabel = label;

    if reset
        return
    end
end

count = count+1;
fprintf('  %s completed %d / %d\n',label,count,total)

end

function a = wrapPi(a)

a = atan2(sin(a),cos(a));

end
