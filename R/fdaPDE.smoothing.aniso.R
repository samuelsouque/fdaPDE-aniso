#' aniso.smooth.FEM.PDE.basis
#' @description Spatial regression with differential regularization: The anisotropy matrix is estimated from data. This function is a wrapper for a C++ execution of the algorithm.
#'
#' @param locations A matrix where each line contains the 2D coordinates of the observations
#' @param observations A vector of length \code{dim(FEMbasis$mesh$nodes)[1]} containing all observations, then \code{NA} until the required length is reached
#' @param FEMbasis An FEM basis oject
#' @param lambda a vector of regularization coefficients used for the estimation of K
#' @param PDE_parameters A list of parameters for the regularizing PDE. Contains K, b and c
#' @param covariates not implemented
#' @param BC not implemented
#'
#' @return a list containing the fit, the misfit and the esstimated anisotropy matrix Kappa
#'
#' @examples
#' #import the data from package fdaPDE
#' data("MeuseData")
#' data("MeuseBorder")
#' # Create and refine a mesh
#' mesh <- create.MESH.2D(nodes = MeuseData[,c(2,3)], segments = MeuseBorder, order = 1)
#' mesh <- refine.MESH.2D(mesh, maximum_area=6000, delaunay=TRUE)
#' #Call the anisotropic regression function
#' smoothing_aniso <- aniso.smooth.FEM.PDE.basis(locations = MeuseData[,2:3], 
#'                                               observations = c(MeuseData$zinc,rep(NA,dim(mesh$nodes)[1]-length(MeuseData$zinc))),
#'                                               FEMbasis = create.FEM.basis(mesh), 
#'                                               lambda = 10^seq(1,10,2), 
#'                                               rho = c(0.01,0.01,0.5,0.9), 
#'                                               PDE_parameters = list(K = matrix(0, 2, 2), b = c(0, 0), c = 0))
#' # Display the fitted function: An anisotropy is visible
#' plot(smoothing_aniso$fit.FEM)
#' 
#' 
aniso.smooth.FEM.PDE.basis <- function(locations = NULL, observations, FEMbasis, lambda, PDE_parameters, covariates = NULL, BC = NULL, GCV = FALSE, CPP_CODE = TRUE, GCVmethod = 2, nrealizations = 100) {
  if (class(FEMbasis$mesh) == "MESH2D") {
    ndim = 2
    mydim = 2
  } else if(class(FEMbasis$mesh) == "MESH.2.5D" || class(FEMbasis$mesh) == "MESH.3D") {
    stop('Function not yet implemented for this mesh class')
  } else {
    stop('Unknown mesh class')
  }
  ##################### Checking parameters, sizes and conversion ################################
  checkSmoothingParameters(locations, observations, FEMbasis, lambda, covariates, BC, GCV, CPP_CODE, PDE_parameters_constant = PDE_parameters, PDE_parameters_func = NULL, GCVmethod , nrealizations)
  
  ## Coverting to format for internal usage
  if (!is.null(locations)) {
    locations = as.matrix(locations)
  }
  observations = as.matrix(observations)
  lambda = as.matrix(lambda)
  if(!is.null(covariates))
    covariates = as.matrix(covariates)
  if(!is.null(BC))
  {
    BC$BC_indices = as.matrix(BC$BC_indices)
    BC$BC_values = as.matrix(BC$BC_values)
  }
  
  if(!is.null(PDE_parameters))
  {
    PDE_parameters$K = as.matrix(PDE_parameters$K)
    PDE_parameters$b = as.matrix(PDE_parameters$b)
    PDE_parameters$c = as.matrix(PDE_parameters$c)
  }
  
  checkSmoothingParametersSize(locations, observations, FEMbasis, lambda, covariates, BC, GCV, CPP_CODE, PDE_parameters_constant = PDE_parameters, PDE_parameters_func = NULL, ndim, mydim)
  ################## End of checking parameters, sizes and conversion #############################
  
  bigsol <- NULL
  
  if(CPP_CODE == FALSE) {
    print('Function implemented only in C++, turn CPP_CODE = TRUE')  
  } else {
    if (GCV) {
      message('Anisotropic smoothing called with GCV = TRUE leads to useless calculations')
    }
    print('C++ Code Execution')
    bigsol = CPP_aniso.smooth.FEM.PDE.basis(locations, observations, FEMbasis, lambda, PDE_parameters, covariates, ndim, mydim, BC, GCV, GCVmethod, nrealizations)
  }
  
  if (is.null(bigsol)) {
    return(bigsol)
  }
  
  numnodes = nrow(FEMbasis$mesh$nodes)
  
  f = bigsol[[1]][1:numnodes,]
  g = bigsol[[1]][(numnodes+1):(2*numnodes),]
  
  # Make Functional objects object
  fit.FEM  = FEM(f, FEMbasis)
  PDEmisfit.FEM = FEM(g, FEMbasis)
  
  # beta = getBetaCoefficients(locations, observations, fit.FEM, covariates, TRUE)
  # reslist=list(fit.FEM=fit.FEM,PDEmisfit.FEM=PDEmisfit.FEM, beta = beta, kappa = bigsol[[2]])
  reslist=list(fit.FEM=fit.FEM,PDEmisfit.FEM=PDEmisfit.FEM, kappa = bigsol[[2]])
  return(reslist)
}

aniso.smooth.FEM.PDE.sv.basis <- function(locations = NULL, observations, FEMbasis, lambda, PDE_parameters, covariates = NULL, BC = NULL, GCV = FALSE, CPP_CODE = TRUE, GCVmethod = 2, nrealizations = 100) {
  if (class(FEMbasis$mesh) == "MESH2D") {
    ndim = 2
    mydim = 2
  } else if (class(FEMbasis$mesh) == "MESH.2.5D" || class(FEMbasis$mesh) == "MESH.3D") { 
    stop('Function not yet implemented for this mesh class')
  } else {
    stop('Unknown mesh class')
  }
  ##################### Checking parameters, sizes and conversion ################################
  checkSmoothingParameters(locations, observations, FEMbasis, lambda, covariates, BC, GCV, CPP_CODE, PDE_parameters_constant = NULL, PDE_parameters_func = PDE_parameters,GCVmethod , nrealizations)
  
  ## Coverting to format for internal usage
  if(!is.null(locations))
    locations = as.matrix(locations)
  observations = as.matrix(observations)
  lambda = as.matrix(lambda)
  if(!is.null(covariates))
    covariates = as.matrix(covariates)
  if(!is.null(BC))
  {
    BC$BC_indices = as.matrix(BC$BC_indices)
    BC$BC_values = as.matrix(BC$BC_values)
  }
  
  checkSmoothingParametersSize(locations, observations, FEMbasis, lambda, covariates, BC, GCV, CPP_CODE, PDE_parameters_constant = NULL, PDE_parameters_func = PDE_parameters, ndim, mydim)
  ################## End checking parameters, sizes and conversion #############################
  
  bigsol = NULL
  
  if(CPP_CODE == FALSE) {
    print('Function implemented only in C++, turn CPP_CODE = TRUE')  
  } else {
    if (GCV) {
      message('Anisotropic smoothing called with GCV = TRUE leads to useless calculations')
    }
    print('C++ Code Execution')
    bigsol = CPP_aniso.smooth.FEM.PDE.sv.basis(locations, observations, FEMbasis, lambda, PDE_parameters, covariates, ndim, mydim, BC, GCV, GCVmethod, nrealizations)
  }
  
  if (is.null(bigsol)) {
    return(bigsol)
  }
  
  numnodes = nrow(FEMbasis$mesh$nodes)
  
  f = bigsol[[1]][1:numnodes,]
  g = bigsol[[1]][(numnodes+1):(2*numnodes),]
  
  # Make Functional objects object
  fit.FEM  = FEM(f, FEMbasis)
  PDEmisfit.FEM = FEM(g, FEMbasis)  

  # beta = getBetaCoefficients(locations, observations, fit.FEM, covariates, TRUE)
  # reslist=list(fit.FEM=fit.FEM,PDEmisfit.FEM=PDEmisfit.FEM, beta = beta, kappa=bigsol[[2]])
  reslist=list(fit.FEM=fit.FEM,PDEmisfit.FEM=PDEmisfit.FEM, kappa=bigsol[[2]])
  
  return(reslist)
}
