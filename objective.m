% objective: computes negative log likelihood of entire training
% dataset as a function of the model parameters, x, a vector defined
% as
%
%   x = vec M;
%
% as well as its gradient:
%
%   f(x) = -∑ᵢ log(yᵢ | z, σ², M)
%   g(x) = ∂f/∂x

function [f, g] = objective(x, centered_rest_fluxes, lya_1pzs, ...
          rest_noise_variances, num_forest_lines, all_transition_wavelengths, ...
          all_oscillator_strengths, z_qsos)

  [num_quasars, num_pixels] = size(centered_rest_fluxes);

  k = numel(x) / num_pixels;

  M = reshape(x, [num_pixels, k]);

  f          = 0;
  dM         = zeros(size(M));

  for i = 1:num_quasars
    ind = (~isnan(centered_rest_fluxes(i, :)));

    % Apr 12: directly pass z_qsos in the argument since we don't want
    % zeros in lya_1pzs to mess up the gradients in spectrum_loss
    zqso_1pz = z_qsos(i) + 1;

    [this_f, this_dM, this_dlog_omega, ...
     this_dlog_c_0, this_dlog_tau_0, this_dlog_beta] ...
        = spectrum_loss(centered_rest_fluxes(i, ind)', lya_1pzs(i, ind)', ...
                        rest_noise_variances(i, ind)', M(ind, :), omega2(ind), ...
                        c_0, tau_0, beta, num_forest_lines, all_transition_wavelengths, ...
                        all_oscillator_strengths, zqso_1pz);

    f               = f               + this_f;
    dM(ind, :)      = dM(ind, :)      + this_dM;
  end

  % apply prior for τ₀ (Kim, et al. 2007)
  tau_0_mu    = 0.0023;
  tau_0_sigma = 0.0007;

  dlog_tau_0 = dlog_tau_0 + ...
      tau_0 * (tau_0 - tau_0_mu) / tau_0_sigma^2;

  % apply prior for β (Kim, et al. 2007)
  beta_mu    = 3.65;
  beta_sigma = 0.21;

  dlog_beta = dlog_beta + ...
      beta * (beta - beta_mu) / beta_sigma^2;

  g = [dM(:); dlog_omega(:); dlog_c_0; dlog_tau_0; dlog_beta];

end
