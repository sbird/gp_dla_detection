% objective: computes negative log likelihood of entire training
% dataset as a function of the model parameters, x, a vector defined
% as
%
%   x = [vec M; log ω; log c₉; log τ₀; log β]
%
% as well as its gradient:
%
%   f(x) = -∑ᵢ log(yᵢ | Lyα z, σ², M, ω, c₀, τ₉, β)
%   g(x) = ∂f/∂x

function [f, g] = objective(x, centered_rest_fluxes, lya_1pzs, ...
          rest_noise_variances, num_forest_lines, all_transition_wavelengths, ...
          all_oscillator_strengths)

  [num_quasars, num_pixels] = size(centered_rest_fluxes);

  k = (numel(x) - 3) / num_pixels - 1;

  ind = (1:(num_pixels * k));
  M = reshape(x(ind), [num_pixels, k]);

  ind = (num_pixels * k + 1):(num_pixels * (k + 1));
  log_omega = x(ind);

  log_c_0   = x(end - 2);
  log_tau_0 = x(end - 1);
  log_beta  = x(end);

  omega2 = exp(2 * log_omega);
  c_0    = exp(log_c_0);
  tau_0  = exp(log_tau_0);
  beta   = exp(log_beta);

  f          = 0;
  dM         = zeros(size(M));
  dlog_omega = zeros(size(log_omega));
  dlog_c_0   = 0;
  dlog_tau_0 = 0;
  dlog_beta  = 0;

  for i = 1:num_quasars
    ind = (~isnan(centered_rest_fluxes(i, :)));

    % in objective_lyseries.m the trick is to use the final
    % element of lya_1pzs to acquire 1+zqso; the trick would
    % not work here since lya_1pz is beyond 1 + zqso 
    % (beyond lylimit to lya).
    % Instead, we assume we've already applied an indicator
    % on lya_1pzs, so maximum value of it would be 1 + z_qso
    % get zqso + 1 from lya_1pzs
    zqso_1pz = max(lya_1pzs(i, :));

    [this_f, this_dM, this_dlog_omega, ...
     this_dlog_c_0, this_dlog_tau_0, this_dlog_beta] ...
        = spectrum_loss(centered_rest_fluxes(i, ind)', lya_1pzs(i, ind)', ...
                        rest_noise_variances(i, ind)', M(ind, :), omega2(ind), ...
                        c_0, tau_0, beta, num_forest_lines, all_transition_wavelengths, ...
                        all_oscillator_strengths, zqso_1pz);

    f               = f               + this_f;
    dM(ind, :)      = dM(ind, :)      + this_dM;
    dlog_omega(ind) = dlog_omega(ind) + this_dlog_omega;
    dlog_c_0        = dlog_c_0        + this_dlog_c_0;
    dlog_tau_0      = dlog_tau_0      + this_dlog_tau_0;
    dlog_beta       = dlog_beta       + this_dlog_beta;

  end

  % apply prior for τ₀ (Kim, et al. 2007)
  % change this prior to multi-DLA paper; sigma is just a guess
  tau_0_mu    = 0.000164; %0.0023;
  tau_0_sigma = 0.000050; %0.0007;

  dlog_tau_0 = dlog_tau_0 + ...
      tau_0 * (tau_0 - tau_0_mu) / tau_0_sigma^2;

  % apply prior for β (Kim, et al. 2007)
  % change this prior to multi-DLA paper; sigma is just a guess
  beta_mu    = 5.2714; %3.65;
  beta_sigma = 0.30;   %0.21;

  dlog_beta = dlog_beta + ...
      beta * (beta - beta_mu) / beta_sigma^2;

  g = [dM(:); dlog_omega(:); dlog_c_0; dlog_tau_0; dlog_beta];

end