% process_qsos: run DLA detection algorithm on specified objects

% load QSO model from training release
variables_to_load = {'rest_wavelengths', 'mu', 'M'};
load(sprintf('%s/learned_qso_model_%s',             ...
    processed_directory(training_release), ...
    training_set_name),                    ...
    variables_to_load{:});

% load redshifts from catalog to process
catalog = load(sprintf('%s/zqso_only_catalog', processed_directory(release)));

z_qsos = catalog.z_qsos;
% Generate samples with different quasar redshifts
max_z_qso = max(catalog.z_qsos);
offset_samples_qso = 1:num_zqso_samples;
offset_samples_qso = z_qso_cut + (offset_samples_qso - 1) * (max_z_qso - z_qso_cut)/num_zqso_samples;

bins = 150;
[z_freq, z_bin] = histcounts(z_qsos, [z_qso_cut : ((max(z_qsos) - z_qso_cut) / bins) : max(z_qsos)]);
for i=length(z_freq):-1:1
    z_freq(i) = sum(z_freq(1:i));
end
z_freq = [0 z_freq];
z_freq = z_freq / max(z_freq);
[z_freq, I] = unique(z_freq);
z_bin = z_bin(I);
offset_samples_qso = interp1(z_freq, z_bin, offset_samples_qso);

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
    'all_pixel_mask'};
load(sprintf('%s/preloaded_zqso_only_qsos', processed_directory(release)), ...
    variables_to_load{:});
test_ind = (catalog.filter_flags == 0);
all_wavelengths    =    all_wavelengths(test_ind);
all_flux           =           all_flux(test_ind);
all_noise_variance = all_noise_variance(test_ind);
all_pixel_mask     =     all_pixel_mask(test_ind);
all_thing_ids      =   catalog.thing_ids(test_ind);

z_qsos = catalog.z_qsos(test_ind);

num_quasars = numel(z_qsos);
%load('./test/M.mat');
% preprocess model interpolants
mu_interpolator = ...
    griddedInterpolant(rest_wavelengths,        mu,        'linear');
M_interpolator = ...
    griddedInterpolant({rest_wavelengths, 1:k}, M,         'linear');

% initialize results
sample_log_posteriors  = nan(num_quasars, num_zqso_samples);
z_map                         = nan(num_quasars, 1);
signal_to_noise               = nan(num_quasars, 1);

z_list                   = 1:length(offset_samples_qso);
%Debug output
%all_mus = cell(size(z_list));

fluxes                   = cell(length(z_list), 1);
rest_wavelengths         = cell(length(z_list), 1);

for quasar_ind = 1:num_quasars %quasar list
    tic;
    
    this_wavelengths    =    all_wavelengths{quasar_ind};
    this_flux           =           all_flux{quasar_ind};
    this_noise_variance = all_noise_variance{quasar_ind};
    this_pixel_mask     =     all_pixel_mask{quasar_ind};
        
    for z_list_ind = 1:length(offset_samples_qso) %variant redshift in quasars
        z_qso = offset_samples_qso(z_list_ind);
        
        if mod(z_list_ind, 500) == 0
            fprintf('processing quasar %i of %i, iteration %i (z_QSO = %0.4f) ...\n', ...
                quasar_ind, num_quasars, z_list_ind, z_qso);
        end
        
        %interpolate observations
        max_observed_lambda = observed_wavelengths(max_lambda, z_qso);
        max_observed_lambda = min(max_observed_lambda, max(this_wavelengths));
        min_observed_lambda = observed_wavelengths(min_lambda, z_qso);
        min_observed_lambda = max(min_observed_lambda, min(this_wavelengths));
        if min_observed_lambda > max_observed_lambda
            % If we have no data in the observed range, this sample is maximally unlikely.
            sample_log_posteriors(quasar_ind, z_list_ind) = -1.e99;
            continue;
        end
        rframe_len = 1000;
        vq_range = min_observed_lambda:(max_observed_lambda - min_observed_lambda)/rframe_len:max_observed_lambda;
        vq_range = vq_range';
        this_rest_flux = interp1(this_wavelengths, this_flux, vq_range);
        this_rest_noise_variance = interp1(this_wavelengths, this_noise_variance, vq_range);
        % convert to QSO rest frame
        this_rest_wavelengths = emitted_wavelengths(vq_range, z_qso);
        
        ind = (this_rest_wavelengths >= min_lambda) & ...
            (this_rest_wavelengths <= max_lambda);
        
        %ind = ind & (~this_pixel_mask);
        
        this_rest_wavelengths = this_rest_wavelengths(ind);
        this_rest_flux             =             this_rest_flux(ind);
        this_rest_noise_variance   =   this_rest_noise_variance(ind);
        
        fluxes{z_list_ind} = this_rest_flux;
        rest_wavelengths{z_list_ind} = this_rest_wavelengths;
        
        % interpolate model onto given wavelengths
        this_mu = mu_interpolator( this_rest_wavelengths);
        this_M  =  M_interpolator({this_rest_wavelengths, 1:k});
        %Debug output
        %all_mus{z_list_ind} = this_mu;
        %all_Ms{z_list_ind} = this_M;
       
        sample_log_priors = 0;
        
        sample_log_posteriors(quasar_ind, z_list_ind) = ...
            log_mvnpdf_low_rank(this_rest_flux, this_mu, this_M, this_rest_noise_variance) + sample_log_priors;

        fprintf_debug(' ... log p(D | z_QSO)     : %0.2f\n', ...
            sample_log_posteriors(quasar_ind, z_list_ind));
    end
    this_sample_log = sample_log_posteriors(quasar_ind, :);
    max_log_likelihood = max(this_sample_log);
    
    [~, I] = max(exp(this_sample_log - max_log_likelihood));
    
    z_map(quasar_ind) = offset_samples_qso(I);                                  %MAP estimate
    
    fprintf(' took %0.3fs.\n', toc);
end

% save results
variables_to_save = {'training_release', 'training_set_name', 'offset_samples_qso', 'sample_log_posteriors',
     'max_z_cut', 'z_map', 'z_qsos', 'all_thing_ids'};

filename = sprintf('%s/processed_zqso_only_qsos_%s-%s', ...
    processed_directory(release), ...
    test_set_name, optTag);

save(filename, variables_to_save{:}, '-v7.3');
