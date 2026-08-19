% =========================================================================
% run_matlab.m — Runner MATLAB/Octave do UH-CMA-ES para comparação cruzada
% Reproduz EXATAMENTE o UHCMAES.m, mas instrumentado para exportar, a cada
% geração, os dados necessários à comparação com a versão Python:
%   sísmico do prior, sísmico do melhor indivíduo, fitness, sigma, rel_error.
% Uso:  octave --quiet run_matlab.m
% Lê:   config.json (na pasta corrente) e o .mat apontado por ele.
% Grava: out_matlab.mat
% =========================================================================

clear all; close all; clc;
pkg load signal statistics communications io;

% --- SeReM local (funções baixadas do repositório oficial) ---
addpath(genpath('./SeReM'));
addpath(genpath('./matlab'));   % SeismicModel.m, AkiRichardsCoefficientsMatrix.m

% =========================================================================
% 0. CONFIG
% =========================================================================
cfg = jsondecode(fileread('config.json'));

% seed (reprodutibilidade dentro do MATLAB)
seed = 42;
if isfield(cfg, 'compare') && isfield(cfg.compare, 'seed')
    seed = cfg.compare.seed;
end
rand('seed', seed); randn('seed', seed);

n_gen_compare = 5;
if isfield(cfg, 'compare') && isfield(cfg.compare, 'n_gen_compare')
    n_gen_compare = cfg.compare.n_gen_compare;
end

% =========================================================================
% 1. DADOS E FÍSICA
% =========================================================================
filename = [cfg.files.data_folder cfg.files.input_filename];
eval(["load ", filename]);   % Vp, Vs, Rho, TimeSeis, Snear, Smid, Sfar

nvars = cfg.physics.nvars;
nm = length(Vp);
N = nm * nvars;

theta_raw = cfg.physics.theta_angles;
if iscell(theta_raw), theta_raw = cell2mat(theta_raw); end
if ischar(theta_raw) || isstring(theta_raw), theta_raw = str2num(theta_raw); end
theta = double(theta_raw(:));

dt = TimeSeis(2) - TimeSeis(1);
freq = cfg.physics.wavelet_freq;
ntw = cfg.physics.wavelet_ntw;
[wavelet, ~] = RickerWavelet(freq, dt, ntw);
WaveMat = WaveletMatrix(wavelet, nm, length(theta));
DiffMat = DifferentialMatrix(nm, nvars);

y_obs_real = [Snear; Smid; Sfar];
Nd = length(y_obs_real);

% ---- funções auxiliares (idênticas ao UHCMAES.m) ----
function s = uncertainty_measurement(f_old, f_new, idx_reev, theta)
    lambda = length(f_old);
    all_values = [f_old, f_new];
    [~, sort_idx] = sort(all_values);
    ranks = zeros(1, 2*lambda);
    ranks(sort_idx) = 1:(2*lambda);
    rank_old = ranks(1:lambda); rank_new = ranks(lambda+1:end);
    r1 = rank_old(idx_reev); r2 = rank_new(idx_reev);
    rank_delta = r2 - r1 - sign(r2 - r1);
    limit = (lambda * theta);
    mean_rank_change = mean(abs(rank_delta));
    s = (mean_rank_change - limit) / lambda;
end

function val_n = normalize_data(val, method, p)
    switch method
        case 'linear'
            denom = p.max - p.min; if denom == 0, denom = 1e-6; end
            val_n = (val - p.min) / denom;
        case 'standard'
            denom = p.std; if denom == 0, denom = 1e-6; end
            val_n = (val - p.mean) / denom;
        case 'log'
            val_n = log(val);
        otherwise
            denom = p.max - p.min; if denom == 0, denom = 1e-6; end
            val_n = (val - p.min) / denom;
    end
end

function val = denormalize_data(val_n, method, p)
    switch method
        case 'linear',    val = val_n * (p.max - p.min) + p.min;
        case 'standard',  val = val_n * p.std + p.mean;
        case 'log',       val = exp(val_n);
        otherwise,        val = val_n * (p.max - p.min) + p.min;
    end
end

% =========================================================================
% 2. PRIOR E COVARIÂNCIA
% =========================================================================
nfilt = cfg.physics.prior_filter_order;
cutofffr = cfg.physics.prior_cutoff_freq;
[b, a] = butter(nfilt, cutofffr);
VpPrior = filtfilt(b, a, Vp);
VsPrior = filtfilt(b, a, Vs);
RhoPrior = filtfilt(b, a, Rho);

norm_method = cfg.physics.normalization.method;
p_norm.vp = struct('min', min(Vp), 'max', max(Vp), 'mean', mean(Vp), 'std', std(Vp));
p_norm.vs = struct('min', min(Vs), 'max', max(Vs), 'mean', mean(Vs), 'std', std(Vs));
p_norm.rho = struct('min', min(Rho), 'max', max(Rho), 'mean', mean(Rho), 'std', std(Rho));

Vp_n = normalize_data(Vp, norm_method, p_norm.vp);
Vs_n = normalize_data(Vs, norm_method, p_norm.vs);
Rho_n = normalize_data(Rho, norm_method, p_norm.rho);
VpPrior_n = normalize_data(VpPrior, norm_method, p_norm.vp);
VsPrior_n = normalize_data(VsPrior, norm_method, p_norm.vs);
RhoPrior_n = normalize_data(RhoPrior, norm_method, p_norm.rho);
x_new = [VpPrior_n; VsPrior_n; RhoPrior_n];

corrlength = cfg.physics.correlation_length_factor * dt;
trow = repmat(0:dt:(nm-1)*dt, nm, 1);
tcol = repmat((0:dt:(nm-1)*dt)', 1, nm);
sigmatime = exp(-((trow - tcol) ./ corrlength).^2);
sigma0 = cov([Vp_n, Vs_n, Rho_n]);
if sum(abs(sigma0(:))) < 1e-8
    sigma0 = 1e-4 * eye(3);
else
    sigma0 = sigma0 + 1e-6 * eye(3);
end
C = kron(sigma0, sigmatime);
C = triu(C) + triu(C,1)';
[B, D_eig] = eig(C);
D = sqrt(abs(diag(D_eig)));
D_safe = D; D_safe(D_safe < 1e-12) = 1e-12;
inv_sqrt_C = B * diag(D_safe.^-1) * B';

% =========================================================================
% 3. PARÂMETROS CMA-ES / UH
% =========================================================================
sigma = cfg.cmaes.sigma_initial;
stop_generations = cfg.cmaes.stop_generations;
stop_tol_diversity = cfg.cmaes.stop_tol_diversity;
reg_type = cfg.cmaes.reg_type;

eigeneval = 0; pc = zeros(N,1); ps = zeros(N,1);
lambda = 4 + floor(3 * log(N));
mu = floor(lambda / 2);
weights = log(mu + 1/2) - log(1:mu)';
weights = weights / sum(weights);
mu_eff = sum(weights)^2 / sum(weights.^2);
cc = (4 + mu_eff/N)/(N + 4 + 2*mu_eff/N);
cs = (mu_eff + 2)/(N + mu_eff + 5);
c1 = 2/((N + 1.3)^2 + mu_eff);
c_mu = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff)/((N + 2)^2 + mu_eff));
damps = 1 + 2*max(0, sqrt((mu_eff - 1)/(N + 1)) - 1) + cs;
chiN = sqrt(N)*(1 - 1/(4*N) + 1/(21*N^2));

r_lambda = cfg.uh.r_lambda; theta_uh = cfg.uh.theta_uh;
alpha_t = cfg.uh.alpha_t; alpha_sigma = cfg.uh.alpha_sigma;
cs_uh = cfg.uh.cs_uh; t_eval = cfg.uh.t_eval_initial;
t_min = cfg.uh.t_min; t_max = cfg.uh.t_max; s_bar = 0;
noise_level = cfg.uh.noise_level;
sigma_err_matrix = noise_level * eye(Nd);

% ---- sísmico do PRIOR (para comparação) ----
Y_prior = SeismicModel(VpPrior, VsPrior, RhoPrior, theta, DiffMat, WaveMat, nvars);

% ---- históricos para comparação ----
HIST.gen = []; HIST.fitness = []; HIST.sigma = []; HIST.rel_error = [];
HIST.correlation = []; HIST.t_eval = []; HIST.diversity = [];
Ybest_first = [];   % sísmico do melhor indivíduo da 1ª geração

fprintf('MATLAB: N=%d lambda=%d Nd=%d nm=%d\n', N, lambda, Nd, nm);

% =========================================================================
% 4. LOOP PRINCIPAL
% =========================================================================
generation = 0;
while generation < stop_generations
    generation = generation + 1;
    arx = zeros(N, lambda);
    arfitness_old = zeros(1, lambda);
    n_samples = round(t_eval);

    switch reg_type
        case '5_exp1',       reg_weight = 5 * exp(-0.1 * generation);
        case '5_exp001',     reg_weight = 5 * exp(-0.001 * generation);
        case '1000_exp0001', reg_weight = 1000 * exp(-0.00001 * generation);
        case 'sigma_exp0001',reg_weight = sigma * exp(-0.0001 * generation);
        otherwise,           reg_weight = 5;
    end

    for k = 1:lambda
        arx(:, k) = x_new + sigma * B * (D .* randn(N, 1));
        X_k = arx(:, k);
        Vp_k = denormalize_data(X_k(1:nm), norm_method, p_norm.vp);
        Vs_k = denormalize_data(X_k(nm+1:2*nm), norm_method, p_norm.vs);
        Rho_k = denormalize_data(X_k(2*nm+1:end), norm_method, p_norm.rho);
        try
            Y_pred_k = SeismicModel(Vp_k, Vs_k, Rho_k, theta, DiffMat, WaveMat, nvars);
        catch
            Y_pred_k = zeros(Nd, 1);
        end
        accum_fit = 0;
        for rep = 1:n_samples
            perturb = sqrt(diag(sigma_err_matrix)) .* randn(Nd, 1);
            y_obs_noisy = y_obs_real + perturb;
            accum_fit = accum_fit + sum((y_obs_noisy - Y_pred_k).^2);
        end
        raw_fitness = accum_fit / n_samples;
        prior_term = sum((X_k - [VpPrior_n; VsPrior_n; RhoPrior_n]).^2);
        arfitness_old(k) = raw_fitness + (reg_weight * prior_term);
    end

    % ---- UH ----
    lambda_reev = max(1, floor(r_lambda * lambda));
    idx_reev = randperm(lambda, lambda_reev);
    arfitness_new = arfitness_old;
    for k = idx_reev
        X_k = arx(:, k);
        Vp_k = denormalize_data(X_k(1:nm), norm_method, p_norm.vp);
        Vs_k = denormalize_data(X_k(nm+1:2*nm), norm_method, p_norm.vs);
        Rho_k = denormalize_data(X_k(2*nm+1:end), norm_method, p_norm.rho);
        try
            Y_pred_k = SeismicModel(Vp_k, Vs_k, Rho_k, theta, DiffMat, WaveMat, nvars);
        catch
            Y_pred_k = zeros(Nd, 1);
        end
        accum_fit = 0;
        for rep = 1:n_samples
            perturb = sqrt(diag(sigma_err_matrix)) .* randn(Nd, 1);
            y_obs_noisy = y_obs_real + perturb;
            accum_fit = accum_fit + sum((y_obs_noisy - Y_pred_k).^2);
        end
        raw_fitness = accum_fit / n_samples;
        prior_term = sum((X_k - [VpPrior_n; VsPrior_n; RhoPrior_n]).^2);
        arfitness_new(k) = raw_fitness + (reg_weight * prior_term);
    end

    s_measure = uncertainty_measurement(arfitness_old, arfitness_new, idx_reev, theta_uh);
    s_bar = (1 - cs_uh) * s_bar + cs_uh * s_measure;
    if s_bar > 0
        if t_eval < t_max, t_eval = min(t_eval * alpha_t, t_max);
        else, sigma = sigma * alpha_sigma; end
    elseif s_bar < 0
        t_eval = max(t_eval / alpha_t, t_min);
    end
    arfitness_final = arfitness_old;
    arfitness_final(idx_reev) = (arfitness_old(idx_reev) + arfitness_new(idx_reev)) / 2;

    % ---- seleção e métricas ----
    [arfitness_sorted, arindex] = sort(arfitness_final);
    arx_sorted = arx(:, arindex);
    dists = sqrt(sum((arx - repmat(x_new, 1, lambda)).^2, 1));
    diversity_metric = mean(dists);

    X_best = arx_sorted(:, 1);
    Vp_best = denormalize_data(X_best(1:nm), norm_method, p_norm.vp);
    Vs_best = denormalize_data(X_best(nm+1:2*nm), norm_method, p_norm.vs);
    Rho_best = denormalize_data(X_best(2*nm+1:end), norm_method, p_norm.rho);
    try
        Y_pred_best = SeismicModel(Vp_best, Vs_best, Rho_best, theta, DiffMat, WaveMat, nvars);
    catch
        Y_pred_best = zeros(Nd, 1);
    end
    rel_error = norm(y_obs_real - Y_pred_best) / norm(y_obs_real);
    cc_mat = corrcoef(y_obs_real, Y_pred_best);
    correlation = cc_mat(1, 2);

    HIST.gen(end+1) = generation;
    HIST.fitness(end+1) = arfitness_sorted(1);
    HIST.sigma(end+1) = sigma;
    HIST.rel_error(end+1) = rel_error;
    HIST.correlation(end+1) = correlation;
    HIST.t_eval(end+1) = t_eval;
    HIST.diversity(end+1) = diversity_metric;
    if generation == 1
        Ybest_first = Y_pred_best;
    end

    if mod(generation, 20) == 0
        fprintf('Gen %d: Fit=%.2e | Sigma=%.3f | RelErr=%.4f | Corr=%.4f\n', ...
            generation, arfitness_sorted(1), sigma, rel_error, correlation);
    end

    % ---- atualização CMA-ES ----
    x_old = x_new;
    x_new = arx_sorted(:, 1:mu) * weights;
    ps = (1 - cs)*ps + sqrt(cs*(2-cs)*mu_eff) * inv_sqrt_C * (x_new - x_old)/sigma;
    h_sig = norm(ps)/sqrt(1-(1-cs)^(2*generation))/chiN < 1.4 + 2/(N+1);
    pc = (1 - cc)*pc + h_sig*sqrt(cc*(2-cc)*mu_eff)*(x_new - x_old)/sigma;
    artmp = (1/sigma)*(arx_sorted(:, 1:mu) - repmat(x_old, 1, mu));
    C = (1 - c1 - c_mu)*C + c1*(pc*pc' + (1-h_sig)*cc*(2-cc)*C) + c_mu*artmp*diag(weights)*artmp';
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));

    if generation - eigeneval > lambda/(c1 + c_mu)/N/10
        eigeneval = generation;
        C = triu(C) + triu(C, 1)';
        [B, D_eig] = eig(C);
        D = sqrt(abs(diag(D_eig)));
        D_safe = D; D_safe(D_safe < 1e-12) = 1e-12;
        inv_sqrt_C = B * diag(D_safe.^-1) * B';
    end
end

% =========================================================================
% 5. EXPORTAÇÃO PARA COMPARAÇÃO
% =========================================================================
x_phys_final = [denormalize_data(x_new(1:nm), norm_method, p_norm.vp); ...
                denormalize_data(x_new(nm+1:2*nm), norm_method, p_norm.vs); ...
                denormalize_data(x_new(2*nm+1:end), norm_method, p_norm.rho)];

save('-mat', 'out_matlab.mat', ...
     'Y_prior', 'Ybest_first', 'HIST', 'x_phys_final', ...
     'VpPrior', 'VsPrior', 'RhoPrior', 'y_obs_real', ...
     'N', 'lambda', 'Nd', 'nm', 'seed');
fprintf('MATLAB: salvo out_matlab.mat (gen final=%d, melhor rel_error=%.4f)\n', ...
        generation, min(HIST.rel_error));
