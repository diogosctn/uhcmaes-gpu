% =========================================================================
% Algoritmo UH-CMA-ES para Inversão Sísmica
% Combinação: Lógica UH (Hansen et al.) + Física de Ondas (SeReM)
% =========================================================================

% Comentado para execução em Batch
% clear all; close all; clc;

pkg load communications
pkg load statistics

addpath(genpath('../SeReM/'));

% =========================================================================
% 0. CARREGAMENTO DAS CONFIGURAÇÕES (JSON)
% =========================================================================
fprintf('Lendo arquivo de configuração config.json...\n');
fname = 'config.json';
if exist(fname, 'file')
    raw_json = fileread(fname);
    cfg = jsondecode(raw_json);
else
    error('Arquivo config.json não encontrado.');
end

% =========================================================================
% 1. CARREGAMENTO DE DADOS E CONFIGURAÇÃO FÍSICA
% =========================================================================
fprintf('Carregando dados sísmicos e configurando física...\n');

% Parâmetros de Arquivo e Física Básica
input_file = cfg.files.input_filename;
filename = [cfg.files.data_folder input_file];
eval(["load ", filename]); % Carrega: Vp, Vs, Rho, TimeSeis, Snear, Smid, Sfar

nvars = cfg.physics.nvars;
nm = length(Vp); % size(Snear, 1) + 1;
N = nm * nvars;

theta_raw = cfg.physics.theta_angles;
if iscell(theta_raw)
    theta_raw = cell2mat(theta_raw);
end
if ischar(theta_raw) || isstring(theta_raw)
    theta_raw = str2num(theta_raw);
end
theta = double(theta_raw(:));

% Construção da Wavelet
dt = TimeSeis(2) - TimeSeis(1);
freq = cfg.physics.wavelet_freq;
ntw = cfg.physics.wavelet_ntw;
[wavelet, ~] = RickerWavelet(freq, dt, ntw);
WaveMat = WaveletMatrix(wavelet, nm, length(theta));
DiffMat = DifferentialMatrix(nm, nvars);

% Dados Observados
y_obs_real = [Snear; Smid; Sfar];
Nd = length(y_obs_real);

% =========================================================
% FUNÇÕES AUXILIARES
% =========================================================

function s = uncertainty_measurement(f_old, f_new, idx_reev, theta)
    % Mesma implementação do UHCMAES original
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
            denom = p.max - p.min;
            if denom == 0, denom = 1e-6; end
            val_n = (val - p.min) / denom;
            
        case 'standard'
            denom = p.std;
            if denom == 0, denom = 1e-6; end
            val_n = (val - p.mean) / denom;
            
        case 'log'
            val_n = log(val);
            
        otherwise
            denom = p.max - p.min;
            if denom == 0, denom = 1e-6; end
            val_n = (val - p.min) / denom;
    end
end

function val = denormalize_data(val_n, method, p)
    switch method
        case 'linear'
            val = val_n * (p.max - p.min) + p.min;
        case 'standard'
            val = val_n * p.std + p.mean;
        case 'log'
            val = exp(val_n);
        otherwise
            val = val_n * (p.max - p.min) + p.min;
    end
end

% =========================================================================
% 2. INICIALIZAÇÃO (PRIORS E COVARIÂNCIA)
% =========================================================================
% Parâmetros de Filtragem do Prior
nfilt = cfg.physics.prior_filter_order;
cutofffr = cfg.physics.prior_cutoff_freq;
[b, a] = butter(nfilt, cutofffr);

VpPrior = filtfilt(b, a, Vp);
VsPrior = filtfilt(b, a, Vs);
RhoPrior = filtfilt(b, a, Rho);

% --- NORMALIZAÇÃO ---
norm_method = cfg.physics.normalization.method;
fprintf('Utilizando normalização do tipo: %s\n', norm_method);

% Parâmetros de normalização baseados nos dados lidos (Vp, Vs, Rho)
p_norm.vp = struct('min', min(Vp), 'max', max(Vp), 'mean', mean(Vp), 'std', std(Vp));
p_norm.vs = struct('min', min(Vs), 'max', max(Vs), 'mean', mean(Vs), 'std', std(Vs));
p_norm.rho = struct('min', min(Rho), 'max', max(Rho), 'mean', mean(Rho), 'std', std(Rho));

% Normalizar os dados originais para cálculo da covariância inicial
Vp_n = normalize_data(Vp, norm_method, p_norm.vp);
Vs_n = normalize_data(Vs, norm_method, p_norm.vs);
Rho_n = normalize_data(Rho, norm_method, p_norm.rho);

% Normalizar o Prior
VpPrior_n = normalize_data(VpPrior, norm_method, p_norm.vp);
VsPrior_n = normalize_data(VsPrior, norm_method, p_norm.vs);
RhoPrior_n = normalize_data(RhoPrior, norm_method, p_norm.rho);

x_new = [VpPrior_n; VsPrior_n; RhoPrior_n];

% Matriz de Covariância
corrlength = cfg.physics.correlation_length_factor * dt; % Fator * dt
trow = repmat(0:dt:(nm-1)*dt, nm, 1);
tcol = repmat((0:dt:(nm-1)*dt)', 1, nm);
sigmatime = exp(-((trow - tcol) ./ corrlength).^2);

sigma0 = cov([Vp_n, Vs_n, Rho_n]);

% --- SALVAGUARDA ALGEBRICA (NUGGET EFFECT) ---
% Se o dado for plano, cov() gera uma matriz nula. Injetamos uma covariância 
% mínima para garantir estabilidade e permitir a exploração do espaço de busca.
if sum(abs(sigma0(:))) < 1e-8
    sigma0 = 1e-4 * eye(3); 
else
    sigma0 = sigma0 + 1e-6 * eye(3); % Regularização de Tikhonov contra matrizes mal-condicionadas
end
% ---------------------------------------------

C = kron(sigma0, sigmatime);

% Decomposição Inicial
C = triu(C) + triu(C,1)';
[B, D_eig] = eig(C);
D = sqrt(abs(diag(D_eig)));

% --- PROTEÇÃO CONTRA ELEMENTOS NULOS NA INVERSÃO (0^-1 = Inf) ---
D_safe = D;
D_safe(D_safe < 1e-12) = 1e-12; 
inv_sqrt_C = B * diag(D_safe.^-1) * B';

% =========================================================================
% 3. PARÂMETROS DO ALGORITMO (UH-CMA-ES)
% =========================================================================
% Parâmetros CMA-ES
sigma = cfg.cmaes.sigma_initial;
stop_generations = cfg.cmaes.stop_generations;
stop_tol_diversity = cfg.cmaes.stop_tol_diversity;

reg_type = cfg.cmaes.reg_type;
gen_method = cfg.cmaes.gen_method;

eigeneval = 0;
pc = zeros(N, 1);
ps = zeros(N, 1);

% Cálculo de Lambda (Mantido dinâmico via fórmula, pois depende de N)
lambda = 4 + floor(3 * log(N));
mu = floor(lambda / 2);
weights = log(mu + 1/2) - log(1:mu)';
weights = weights / sum(weights);
mu_eff = sum(weights)^2 / sum(weights.^2);

% Constantes de Adaptação
cc = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N);
cs = (mu_eff + 2) / (N + mu_eff + 5);
c1 = 2 / ((N + 1.3)^2 + mu_eff);
c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((N + 2)^2 + mu_eff));
damps = 1 + 2 * max(0, sqrt((mu_eff - 1) / (N + 1)) - 1) + cs;
chiN = sqrt(N) * (1 - 1/(4*N) + 1/(21*N^2));

% Parâmetros UH (Uncertainty Handling)
r_lambda = cfg.uh.r_lambda;
theta_uh = cfg.uh.theta_uh;
alpha_t = cfg.uh.alpha_t;
alpha_sigma = cfg.uh.alpha_sigma;
cs_uh = cfg.uh.cs_uh;
t_eval = cfg.uh.t_eval_initial;
t_min = cfg.uh.t_min;
t_max = cfg.uh.t_max;
s_bar = 0;

% Controle de Ruído Artificial
noise_level = cfg.uh.noise_level;
sigma_err_matrix = noise_level * eye(Nd);

% Históricos
history.fitness = []; history.solutions = []; history.t_eval = [];
history.sigma = []; history.diversity = []; history.reg_weight = [];
history.ensamble = [];
history.relative_error = [];
history.correlation = [];
history.chi_squared = [];

% =========================================================================
% CONFIGURAÇÃO DE LOGS, PASTAS E BACKUP DE PARÂMETROS
% =========================================================================
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

% 1. Definição da Estrutura de Pastas
base_folder = 'Results_UHCMAES_unisim_2';
run_folder = fullfile(base_folder, timestamp);

% 2. Criação das Pastas (se não existirem)
if ~exist(base_folder, 'dir'), mkdir(base_folder); end
if ~exist(run_folder, 'dir'), mkdir(run_folder); end

% 3. Definição dos Caminhos dos Arquivos de Dados
csv_filename = fullfile(run_folder, 'log_execucao.csv');
mat_filename = fullfile(run_folder, 'run_data.mat');
json_backup_filename = fullfile(run_folder, 'config_used.json');

% 4. Salvar cópia dos parâmetros (JSON) na pasta de resultados
fprintf('Salvando parâmetros da simulação em: %s\n', json_backup_filename);

% Tenta usar PrettyPrint (formatação bonita), se falhar (Octave antigo), usa padrão
try
    json_content = jsonencode(cfg, 'PrettyPrint', true);
catch
    json_content = jsonencode(cfg);
end

fid_json = fopen(json_backup_filename, 'w');
if fid_json == -1
    warning('Não foi possível salvar o arquivo JSON de configuração.');
else
    fprintf(fid_json, '%s', json_content);
    fclose(fid_json);
end

% 5. Cria o cabeçalho do CSV (Log de execução)
fid = fopen(csv_filename, 'w');
fprintf(fid, 'Gen,Fitness,Sigma,Samples,Diversity,S_bar,RegWeight,RelError,Corr,Chi2\n');
fclose(fid);

fprintf('Resultados serão salvos na pasta: %s\n', run_folder);
fprintf('Iniciando Inversão Sísmica com UH-CMA-ES (Dimensão: %d)...\n', N);

% =========================================================================
% 4. LOOP PRINCIPAL
% =========================================================================
generation = 0;
tempo_execucao = 0; % INICIALIZA O ACUMULADOR DE TEMPO DA INVERSÃO

while generation < stop_generations
    tic_iter = tic; % INICIA O CRONÔMETRO DESTA ITERAÇÃO

    generation = generation + 1;

    % --- Passo 1: Amostragem e Avaliação ---
    arx = zeros(N, lambda);
    arfitness_old = zeros(1, lambda);
    n_samples = round(t_eval);

    % Termo de Regularização
    switch reg_type
        case '5_exp1'
            reg_weight = 5 * exp(-0.1 * generation);
        case '5_exp001'
            reg_weight = 5 * exp(-0.001 * generation);

        case '1000_exp0001'
            reg_weight = 1000 * exp(-0.00001 * generation);

        case 'sigma_exp0001'
            reg_weight = sigma * exp(-0.0001 * generation);

        otherwise
            reg_weight = 5;
    end

    for k = 1:lambda
        % Gera indivíduo (Modelo de Velocidade/Densidade)
        switch gen_method
            case 'cmaes'
                arx(:, k) = x_new + sigma * B * (D .* randn(N, 1));

            case 'mvnrnd'
                arx(:, k) = mvnrnd(x_new(:),sigma*C);
        end

        % Decodifica propriedades físicas (Espaço Normalizado)
        X_k = arx(:, k);
        Vp_k_n = X_k(1:nm); Vs_k_n = X_k(nm+1:2*nm); Rho_k_n = X_k(2*nm+1:end);

        % Desnormaliza para o Espaço Físico para o Forward Modeling
        Vp_k = denormalize_data(Vp_k_n, norm_method, p_norm.vp);
        Vs_k = denormalize_data(Vs_k_n, norm_method, p_norm.vs);
        Rho_k = denormalize_data(Rho_k_n, norm_method, p_norm.rho);

        % Forward Modeling (Simulação Física)
        try
            Y_pred_k = SeismicModel(Vp_k, Vs_k, Rho_k, theta, DiffMat, WaveMat, nvars);
        catch
            Y_pred_k = zeros(Nd, 1); % Fallback em caso de erro numérico
        end

        % Avaliação com Resampling (UH Wrapper)
        accum_fit = 0;
        for rep = 1:n_samples
            % Adiciona Ruído aos DADOS OBSERVADOS (Simula medição incerta)
            perturb = sqrt(diag(sigma_err_matrix)) .* randn(Nd, 1);
            y_obs_noisy = y_obs_real + perturb;

            % Misfit (Erro Quadrático)
            misfit = sum((y_obs_noisy - Y_pred_k).^2);
            accum_fit = accum_fit + misfit;
        end
        raw_fitness = accum_fit / n_samples;

        % Adiciona Regularização (Mantém consistência geológica)
        % Penaliza desvios extremos do modelo a priori (No espaço normalizado)
        prior_term = sum((X_k - [VpPrior_n; VsPrior_n; RhoPrior_n]).^2);
        arfitness_old(k) = raw_fitness + (reg_weight * prior_term);
    end

    % --- Passo 2: Uncertainty Handling (UH) ---
    % Seleciona para reavaliação
    lambda_reev = max(1, floor(r_lambda * lambda));
    idx_reev = randperm(lambda, lambda_reev);
    arfitness_new = arfitness_old;

    for k = idx_reev
        X_k = arx(:, k);
        Vp_k_n = X_k(1:nm); Vs_k_n = X_k(nm+1:2*nm); Rho_k_n = X_k(2*nm+1:end);

        % Desnormaliza para o Espaço Físico
        Vp_k = denormalize_data(Vp_k_n, norm_method, p_norm.vp);
        Vs_k = denormalize_data(Vs_k_n, norm_method, p_norm.vs);
        Rho_k = denormalize_data(Rho_k_n, norm_method, p_norm.rho);

        try
            Y_pred_k = SeismicModel(Vp_k, Vs_k, Rho_k, theta, DiffMat, WaveMat, nvars);
        catch
            Y_pred_k = zeros(Nd, 1);
        end

        accum_fit = 0;
        for rep = 1:n_samples
            % NOVO ruído aleatório para testar estabilidade
            perturb = sqrt(diag(sigma_err_matrix)) .* randn(Nd, 1);
            y_obs_noisy = y_obs_real + perturb;
            misfit = sum((y_obs_noisy - Y_pred_k).^2);
            accum_fit = accum_fit + misfit;
        end
        raw_fitness = accum_fit / n_samples;
        prior_term = sum((X_k - [VpPrior_n; VsPrior_n; RhoPrior_n]).^2);
        arfitness_new(k) = raw_fitness + (reg_weight * prior_term);
    end

    % Medição e Tratamento de Incerteza
    s_measure = uncertainty_measurement(arfitness_old, arfitness_new, idx_reev, theta_uh);
    s_bar = (1 - cs_uh) * s_bar + cs_uh * s_measure;

    if s_bar > 0
        if t_eval < t_max
            t_eval = min(t_eval * alpha_t, t_max);
        else
            sigma = sigma * alpha_sigma;
        end
    elseif s_bar < 0
        t_eval = max(t_eval / alpha_t, t_min);
    end

    % Agregação Final
    arfitness_final = arfitness_old;
    arfitness_final(idx_reev) = (arfitness_old(idx_reev) + arfitness_new(idx_reev)) / 2;

    % --- Passo 3: Atualização CMA-ES ---
    [arfitness_sorted, arindex] = sort(arfitness_final);
    arx_sorted = arx(:, arindex);

    % Métricas de Diversidade
    dists = sqrt(sum((arx - repmat(x_new, 1, lambda)).^2, 1));
    diversity_metric = mean(dists);

    % Salva Histórico Base
    history.fitness = [history.fitness; arfitness_sorted(1)];
    history.t_eval = [history.t_eval; t_eval];
    history.sigma = [history.sigma; sigma];
    history.diversity = [history.diversity; diversity_metric];
    history.reg_weight = [history.reg_weight; reg_weight];

    % --- CÁLCULO DAS MÉTRICAS DE RESÍDUO (MELHOR INDIVÍDUO) ---
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
    chi_squared = mean((y_obs_real - Y_pred_best).^2) / noise_level^2;

    history.relative_error = [history.relative_error; rel_error];
    history.correlation = [history.correlation; correlation];
    history.chi_squared = [history.chi_squared; chi_squared];

    % --- CRITÉRIO DE PARADA CONFIGURÁVEL ---
    stop_method = cfg.cmaes.stop_criteria.method;
    stop_threshold = cfg.cmaes.stop_criteria.threshold;
    stop_now = false;

    switch stop_method
        case 'relative_error'
            if rel_error < stop_threshold
                stop_now = true;
                fprintf('>>> Parada: Erro Relativo (%.4f) < Tol (%.4f)\n', rel_error, stop_threshold);
            end
        case 'correlation'
            if correlation > stop_threshold
                stop_now = true;
                fprintf('>>> Parada: Correlação (%.4f) > Tol (%.4f)\n', correlation, stop_threshold);
            end
        case 'chi_squared'
            if chi_squared < stop_threshold
                stop_now = true;
                fprintf('>>> Parada: Qui-Quadrado (%.4f) < Tol (%.4f)\n', chi_squared, stop_threshold);
            end
        case 'stagnation'
            window = cfg.cmaes.stop_criteria.stagnation_window;
            if generation > window
                prev_fit = history.fitness(end-window);
                curr_fit = history.fitness(end);
                improvement = abs(prev_fit - curr_fit) / abs(prev_fit);
                if improvement < stop_threshold
                    stop_now = true;
                    fprintf('>>> Parada: Estagnação (Melhoria %.2e) < Tol (%.2e)\n', improvement, stop_threshold);
                end
            end
        case 'diversity'
            if diversity_metric < stop_tol_diversity
                stop_now = true;
                fprintf('>>> Parada: Diversidade (%.4f) < Tol (%.4f)\n', diversity_metric, stop_tol_diversity);
            end
    end

    if stop_now
        tempo_execucao = tempo_execucao + toc(tic_iter);
        break;
    end

    % Atualização dos Parâmetros de Distribuição
    x_old = x_new;
    x_new = arx_sorted(:, 1:mu) * weights;

    ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * inv_sqrt_C * (x_new - x_old) / sigma;
    h_sig = norm(ps) / sqrt(1 - (1 - cs)^(2 * generation)) / chiN < 1.4 + 2 / (N + 1);
    pc = (1 - cc) * pc + h_sig * sqrt(cc * (2 - cc) * mu_eff) * (x_new - x_old) / sigma;

    artmp = (1/sigma) * (arx_sorted(:, 1:mu) - repmat(x_old, 1, mu));
    C = (1 - c1 - c_mu) * C + c1 * (pc * pc' + (1 - h_sig) * cc * (2 - cc) * C) + c_mu * artmp * diag(weights) * artmp';

    sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1));

    % Decomposição
    if generation - eigeneval > lambda / (c1 + c_mu) / N / 10
        eigeneval = generation;
        C = triu(C) + triu(C, 1)';
        [B, D_eig] = eig(C);
        D = sqrt(abs(diag(D_eig)));
        
        % --- REPLIQUE A PROTEÇÃO AQUI DENTRO DO LOOP ---
        D_safe = D;
        D_safe(D_safe < 1e-12) = 1e-12;
        inv_sqrt_C = B * diag(D_safe.^-1) * B';
        % -----------------------------------------------
    end

    % PÁRA O CRONÔMETRO AQUI, ANTES DOS PROCESSOS DE I/O (Prints e Gravação em Disco)
    tempo_execucao = tempo_execucao + toc(tic_iter);

    if mod(generation, 25) == 0
        % Desnormaliza para salvar os resultados em unidades físicas
        x_phys = [denormalize_data(x_new(1:nm), norm_method, p_norm.vp); ...
                  denormalize_data(x_new(nm+1:2*nm), norm_method, p_norm.vs); ...
                  denormalize_data(x_new(2*nm+1:end), norm_method, p_norm.rho)];

        ensamble_phys = zeros(size(arx));
        for i_ens = 1:size(arx, 2)
            ensamble_phys(:, i_ens) = [denormalize_data(arx(1:nm, i_ens), norm_method, p_norm.vp); ...
                                       denormalize_data(arx(nm+1:2*nm, i_ens), norm_method, p_norm.vs); ...
                                       denormalize_data(arx(2*nm+1:end, i_ens), norm_method, p_norm.rho)];
        end

        history.ensamble = ensamble_phys;
        history.solutions(:, end+1) = x_phys;
        history.time_elapsed = tempo_execucao; % Salva o tempo atualizado no history

        fprintf('Gen %d: Fit=%.2e | Sigma=%.2f | Div=%.2f | RelErr=%.4f | Corr=%.4f | Chi2=%.2f\n', ...
            generation, arfitness_sorted(1), sigma, diversity_metric, rel_error, correlation, chi_squared);

        % --- SALVAR CSV ---
        fid = fopen(csv_filename, 'a');
        fprintf(fid, '%d,%.6e,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
            generation, arfitness_sorted(1), sigma, round(t_eval), ...
            diversity_metric, s_bar, reg_weight, rel_error, correlation, chi_squared);
        fclose(fid);

        % --- SALVAR MAT ---
        x_new_norm = x_new; % Preserva o estado normalizado para o loop
        x_new = x_phys;     % Temporariamente muda para o físico para o save
        save(mat_filename, 'history', 'x_new', 'cfg');
        x_new = x_new_norm; % Restaura o estado normalizado
    end
end

% =========================================================================
% 5. SALVAMENTO DEFINITIVO APÓS O FIM DO LOOP
% =========================================================================

% Desnormaliza a solução final (centro da distribuição)
x_phys_final = [denormalize_data(x_new(1:nm), norm_method, p_norm.vp); ...
                denormalize_data(x_new(nm+1:2*nm), norm_method, p_norm.vs); ...
                denormalize_data(x_new(2*nm+1:end), norm_method, p_norm.rho)];

% Atualiza o Ensemble e as Soluções finais no histórico (caso não tenha sido salvo no último ciclo do loop)
if exist('arx', 'var') && (mod(generation, 25) ~= 0)
    % Desnormaliza o ensemble final
    ensamble_phys_final = zeros(size(arx));
    for i_ens = 1:size(arx, 2)
        ensamble_phys_final(:, i_ens) = [denormalize_data(arx(1:nm, i_ens), norm_method, p_norm.vp); ...
                                         denormalize_data(arx(nm+1:2*nm, i_ens), norm_method, p_norm.vs); ...
                                         denormalize_data(arx(2*nm+1:end, i_ens), norm_method, p_norm.rho)];
    end

    % Atualiza a estrutura de histórico
    history.ensamble = ensamble_phys_final;
    history.solutions(:, end+1) = x_phys_final;

    % Grava a última linha no CSV
    fid = fopen(csv_filename, 'a');
    fprintf(fid, '%d,%.6e,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        generation, arfitness_sorted(1), sigma, round(t_eval), ...
        diversity_metric, s_bar, reg_weight, rel_error, correlation, chi_squared);
    fclose(fid);
end

history.time_elapsed = tempo_execucao; % Garante que o tempo final absoluto esteja gravado
x_new = x_phys_final; % Sobrescreve a variável principal para o save final
save(mat_filename, 'history', 'x_new', 'cfg');

fprintf('Execução estrita do algoritmo UHCMAES concluída na geração %d em %.2f segundos.\n', generation, tempo_execucao);
fprintf('Processo finalizado com sucesso!\n');
