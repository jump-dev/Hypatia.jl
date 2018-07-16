

export AlfonsoOptimizer

mutable struct AlfonsoOptimizer <: MOI.AbstractOptimizer



end





if ~isfield(opts, 'predLineSearch'); opts.predLineSearch = 1; end;
if ~isfield(opts, 'maxCorrSteps'); opts.maxCorrSteps = 4; end;
if ~isfield(opts, 'corrCheck'); opts.corrCheck = 1; end;
if ~isfield(opts, 'optimTol'); opts.optimTol = 1e-06; end;
if ~isfield(opts, 'maxCorrLSIters'); opts.maxCorrLSIters = 8; end;
if ~isfield(opts, 'maxPredSmallSteps'); opts.maxPredSmallSteps = 8; end;
if ~isfield(opts, 'maxItRefineSteps'); opts.maxItRefineSteps = 0; end;
if ~isfield(opts, 'verbose'); opts.verbose = 1; end;



function algParams = setAlgParams(gH_Params, opts)
% This method sets the algorithmic parameters.
% --------------------------------------------------------------------------
% USAGE of "setAlgParams"
% algParams = setAlgParams(gH_Params, opts)
% --------------------------------------------------------------------------
% INPUT
% gH_Params:	parameters associated with the method gH
% opts:         algorithmic options
%
% OUTPUT
% algParams:                        algorithmic parameters
% - algParams.maxIter:              maximum number of iterations
% - algParams.optimTol:               optimization tolerance parameter
% - algParams.alphaCorr:            corrector step size
% - algParams.predLSMulti:          predictor line search step size
%                                   multiplier
% - algParams.corrLSMulti:          corrector line search step size
%                                   multiplier
% - algParams.itRefineThreshold:    iterative refinement success threshold
% - algParams.maxCorrSteps:         maximum number of corrector steps
% - algParams.beta:                 large neighborhood parameter
% - algParams.eta:                  small neighborhood parameter
% - algParams.alphaPredLS:          initial predictor step size with line
%                                   search
% - algParams.alphaPredFix:         fixed predictor step size
% - algParams.alphaPred:            initial predictor step size
% - algParams.alphaPredThreshold:   minimum predictor step size
% --------------------------------------------------------------------------
% EXTERNAL FUNCTIONS CALLED IN THIS FUNCTION
% None.
% --------------------------------------------------------------------------

    algParams.maxIter           = 10000;
    algParams.optimTol            = opts.optimTol;

    algParams.alphaCorr         = 1.0;
    algParams.predLSMulti       = 0.7;
    algParams.corrLSMulti       = 0.5;
    algParams.itRefineThreshold = 0.1;

    % parameters are chosen to make sure that each predictor
    % step takes the current iterate from the eta-neighborhood to the
    % beta-neighborhood and each corrector phase takes the current
    % iterate from the beta-neighborhood to the eta-neighborhood.
    % extra corrector steps are allowed to mitigate the effects of
    % finite precision.
    algParams.maxCorrSteps      = 2*opts.maxCorrSteps;

    % precomputed safe parameters
    switch opts.maxCorrSteps
        case 1
            if gH_Params.bnu < 10
                algParams.beta       = 0.1810;
                algParams.eta        = 0.0733;
                cPredFix             = 0.0225;
            elseif gH_Params.bnu < 100
                algParams.beta       = 0.2054;
                algParams.eta        = 0.0806;
                cPredFix             = 0.0263;
            else
                algParams.beta       = 0.2190;
                algParams.eta        = 0.0836;
                cPredFix             = 0.0288;
            end
        case 2
            if gH_Params.bnu < 10
                algParams.beta       = 0.2084;
                algParams.eta        = 0.0502;
                cPredFix             = 0.0328;
            elseif gH_Params.bnu < 100
                algParams.beta       = 0.2356;
                algParams.eta        = 0.0544;
                cPredFix             = 0.0380;
            else
                algParams.beta       = 0.2506;
                algParams.eta        = 0.0558;
                cPredFix             = 0.0411;
            end
        case 4
            if gH_Params.bnu < 10
                algParams.beta       = 0.2387;
                algParams.eta        = 0.0305;
                cPredFix             = 0.0429;
            elseif gH_Params.bnu < 100
                algParams.beta       = 0.2683;
                algParams.eta        = 0.0327;
                cPredFix             = 0.0489;
            else
                algParams.beta       = 0.2844;
                algParams.eta        = 0.0332;
                cPredFix             = 0.0525;
            end
        otherwise
            error('The maximum number of corrector steps can be 1, 2, or 4.');
    end

    kx = algParams.eta + sqrt(2*algParams.eta^2 + gH_Params.bnu);
    algParams.alphaPredFix  = cPredFix/kx;
    algParams.alphaPredLS   = min(100 * algParams.alphaPredFix, 0.9999);
    algParams.alphaPredThreshold = (algParams.predLSMulti^opts.maxPredSmallSteps)*algParams.alphaPredFix;

    if opts.predLineSearch == 0
        % fixed predictor step size
        algParams.alphaPred   = algParams.alphaPredFix;
    else
        % initial predictor step size with line search
        algParams.alphaPred   = algParams.alphaPredLS;
    end

return
