function Z = rmsNormalization(X, g, epsilon)
% rmsNormalization   Root Mean Square Normalization
%
%   Z = rmsNormalization(X, g, epsilon) applies RMS normalization to the input X.
%   Used in models like Llama, Gopher, etc.
%
%   Inputs:
%       X       - A numFeatures-by-numInputSubwords-by-numObs input array.
%       g       - A numFeatures-by-1 weight vector (gamma).
%       epsilon - Small constant for numerical stability (default 1e-6).
%
%   Outputs:
%       Z       - A numFeatures-by-numInputSubwords-by-numObs output array.

if nargin < 3
    epsilon = 1e-6;
end

normalizationDimension = 1;

% Ensure scaling vector g is a column vector aligned with first dimension of X
if isrow(g)
    g = g(:);
end

% Check compatibility
if size(g,1) ~= size(X,1)
     error('transformer:rmsNormalization:DimensionMismatch', ...
        'Scale vector g length (%d) must match input feature dimension (%d).', size(g,1), size(X,1));
end

% Calculate RMS
% Mean of squares along feature dimension
ms = mean(X.^2, normalizationDimension);
rms = sqrt(ms + epsilon);

% Normalize and scale
% X ./ rms broadcasts along dim 1
Z = (X ./ rms) .* g;

end
