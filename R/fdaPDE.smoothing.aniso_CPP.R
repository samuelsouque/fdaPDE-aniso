CPP_aniso.smooth.FEM.PDE.basis <- function(locations, observations, FEMbasis, lambda, PDE_parameters, covariates, ndim, mydim, BC, GCV, GCVmethod, nrealizations) {
  # Indexes in C++ starts from 0, in R from 1, opportune transformation  
  ##TO BE CHANGED SOON: LOW PERFORMANCES, IMPLIES COPY OF PARAMETERS
  FEMbasis$mesh$triangles = FEMbasis$mesh$triangles - 1
  FEMbasis$mesh$edges = FEMbasis$mesh$edges - 1
  FEMbasis$mesh$neighbors[FEMbasis$mesh$neighbors != -1] = FEMbasis$mesh$neighbors[FEMbasis$mesh$neighbors != -1] - 1
  
  if (is.null(covariates)) {
    covariates<-matrix(nrow = 0, ncol = 1)
  }
  
  if (is.null(locations)) {
    locations<-matrix(nrow = 0, ncol = 2)
  }
  
  if (is.null(BC$BC_indices)) {
    BC$BC_indices<-vector(length=0)
  } else {
    BC$BC_indices<-as.vector(BC$BC_indices)-1
  }
  
  if (is.null(BC$BC_values)) {
    BC$BC_values<-vector(length=0)
  } else {
    BC$BC_values<-as.vector(BC$BC_values)
  }
  
  ## Set propr type for correct C++ reading
  locations <- as.matrix(locations)
  storage.mode(locations) <- "double"
  storage.mode(FEMbasis$mesh$points) <- "double"
  storage.mode(FEMbasis$mesh$triangles) <- "integer"
  storage.mode(FEMbasis$mesh$edges) <- "integer"
  storage.mode(FEMbasis$mesh$neighbors) <- "integer"
  storage.mode(FEMbasis$order) <- "integer"
  covariates = as.matrix(covariates)
  storage.mode(covariates) <- "double"
  storage.mode(ndim) <- "integer"
  storage.mode(mydim) <- "integer"
  storage.mode(lambda)<- "double"
  storage.mode(BC$BC_indices)<- "integer"
  storage.mode(BC$BC_values)<-"double"
  
  GCV = as.integer(GCV)
  storage.mode(GCV) <- "integer"
  
  storage.mode(PDE_parameters$K) <- "double"
  storage.mode(PDE_parameters$b) <- "double"
  storage.mode(PDE_parameters$c) <- "double"
  
  storage.mode(nrealizations) <- "integer"
  storage.mode(GCVmethod) <- "integer"
  
  ## Call C++ function
  bigsol <- .Call("anisotropic_regression_PDE", locations, observations, FEMbasis$mesh, FEMbasis$order, mydim, ndim, lambda, PDE_parameters$K, PDE_parameters$b, PDE_parameters$c, covariates, BC$BC_indices, BC$BC_values, GCV, GCVmethod, nrealizations, PACKAGE = "fdaPDE")
  
  return(bigsol)
}

CPP_aniso.smooth.FEM.PDE.sv.basis<-function(locations, observations, FEMbasis, lambda, PDE_parameters, covariates = NULL, ndim, mydim, BC = NULL, GCV, GCVmethod, nrealizations)
{
  # Indexes in C++ starts from 0, in R from 1, opportune transformation
  ##TO BE CHANGED SOON: LOW PERFORMANCES, IMPLIES COPY OF PARAMETERS
  FEMbasis$mesh$triangles = FEMbasis$mesh$triangles - 1
  FEMbasis$mesh$edges = FEMbasis$mesh$edges - 1
  FEMbasis$mesh$neighbors[FEMbasis$mesh$neighbors != -1] = FEMbasis$mesh$neighbors[FEMbasis$mesh$neighbors != -1] - 1
  
  if (is.null(covariates))
  {
    covariates <- matrix(nrow = 0, ncol = 1)
  }
  
  if (is.null(locations))
  {
    locations <- matrix(nrow = 0, ncol = 2)
  }
  
  if (is.null(BC$BC_indices)) {
    BC$BC_indices <- vector(length=0)
  } else {
    BC$BC_indices <- as.vector(BC$BC_indices)-1
  }
  
  if (is.null(BC$BC_values)) {
    BC$BC_values <- vector(length=0)
  } else {
    BC$BC_values <- as.vector(BC$BC_values)
  }
  
  PDE_param_eval = NULL
  points_eval = matrix(CPP_get_evaluations_points(mesh = FEMbasis$mesh, order = FEMbasis$order),ncol = 2)
  PDE_param_eval$K = (PDE_parameters$K)(points_eval)
  PDE_param_eval$b = (PDE_parameters$b)(points_eval)
  PDE_param_eval$c = (PDE_parameters$c)(points_eval)
  PDE_param_eval$u = (PDE_parameters$u)(points_eval)
  
  ## Set propr type for correct C++ reading
  locations <- as.matrix(locations)
  storage.mode(locations) <- "double"
  storage.mode(FEMbasis$mesh$nodes) <- "double"
  storage.mode(FEMbasis$mesh$triangles) <- "integer"
  storage.mode(FEMbasis$mesh$edges) <- "integer"
  storage.mode(FEMbasis$mesh$neighbors) <- "integer"
  storage.mode(FEMbasis$order) <- "integer"
  covariates = as.matrix(covariates)
  storage.mode(covariates) <- "double"
  storage.mode(ndim) <- "integer"
  storage.mode(mydim) <- "integer"
  storage.mode(lambda) <- "double"
  storage.mode(BC$BC_indices) <- "integer"
  storage.mode(BC$BC_values) <-"double"
  storage.mode(GCV) <-"integer"
  
  storage.mode(PDE_param_eval$K) <- "double"
  storage.mode(PDE_param_eval$b) <- "double"
  storage.mode(PDE_param_eval$c) <- "double"
  storage.mode(PDE_param_eval$u) <- "double"
  
  storage.mode(nrealizations) <- "integer"
  storage.mode(GCVmethod) <- "integer"
  
  ## Call C++ function
  bigsol <- .Call("anisotropic_regression_PDE_space_varying", locations, observations, FEMbasis$mesh, FEMbasis$order,mydim, ndim, lambda, PDE_param_eval$K, PDE_param_eval$b, PDE_param_eval$c, PDE_param_eval$u, covariates, BC$BC_indices, BC$BC_values, GCV, GCVmethod, nrealizations, PACKAGE = "fdaPDE")
  return(bigsol)
}