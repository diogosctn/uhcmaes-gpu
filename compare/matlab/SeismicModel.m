function Seis = SeismicModel (Vp, Vs, Rho, theta, DiffMat, WaveMat, nv)

% SEISMIC MODEL computes synthetic seismic data according to a *linearized*
% seismic model based on the convolution of a wavelet and the *linearized*
% approximation of Zoeppritz equations
% INPUT Vp = P-wave velocity profile
%       Vs = S-wave velocity profile
%       Rho = Density profile
%       DiffMat = differential matrix (pre-calc)
%       WaveMat = wavelet matrix (pre-calc)
%       nv = number of variables (=3)
% OUTUPT Seis = vector of seismic data of size (nsamples x nangles, 1)

% Written by Dario Grana (August 2020)

  % From ground data (Vp, Vs, Rho) to seismic signal:
  m = [log(Vp); log(Vs); log(Rho)]; % taking logs
  mder = DiffMat * m;               % differentials of ground model
  A = AkiRichardsCoefficientsMatrix(Vp, Vs, theta, nv);
  Cpp = A * mder;                   % converting to reflectivities
  Seis = WaveMat * Cpp;             % converting to seismic data signals

endfunction
