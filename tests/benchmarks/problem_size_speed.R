# Benchmark: Study speed as a functino of problem size
# For each random field and for each size n in a predetermined range, we select randomly n points, solve the system and write down the time.
setwd("~/PoliMi/PACS")
library(fdaPDE)


# dominio
lato <- 5
x_L <- y_L <- -lato
x_U <- y_U <- lato
limite <- 1
len_bordo <- length(seq(from=x_L,to=x_U,by=limite))
boundary <- cbind(c(seq(from=x_L,to=x_U,by=limite)[-1],rep(x_U,len_bordo)[-1],seq(from=x_U,to=x_L,by=-limite)[-1],rep(x_L,len_bordo)[-1]),
                  c(rep(y_L,len_bordo)[-1],seq(from=y_L,to=y_U,by=limite)[-1],rep(y_U,len_bordo)[-1],seq(from=y_U,to=y_L,by=-limite)[-1]))
lato <- 5
n <- length(data) # numero di dati
omega <- (lato*2)^2 # area dominio

# boundary indices
bdd <- which(data[,1] %in% boundary[,1] * 
               data[,2] %in% boundary[,2] == 1)
bdd <- bdd[c(1:11,12,22,23,33,34,44,45,55,56,66,67,77,78,88,89,99,100,110:121)]

# param ####
rhovec <- c(0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)

# ####
data_size <- c(seq(100,1000,100),seq(2000,10000,1000),seq(12000,40000,2000))
data_size <- c(seq(100,1000,100),seq(1500,3000,500)) # we go until 3000, because tests gave the conclusion that more takes too much time
time <- matrix(rep(NA,length(data_size)*5), ncol = 5)

# ####
iterazione <- 1

data <- read.table(paste(getwd(),"/random_fields/random_field_test_",iterazione,".txt",sep=''),  header=FALSE)
D <- dim(data)

# select boundary
data[bdd,]
plot(data[bdd,1],data[bdd,2])

# ####
#TODO
i=100
for(i in 1:length(data_size)){
  
  
  indices_i <- sample(setdiff(1:D[1],bdd), data_size[i], replace = FALSE)
  data_i <- data[c(indices_i),]
  #plot(data_i[,1], data_i[,2])
  
  data_locations <- matrix(NA,nrow=dim(data_i)[1],ncol=2)
  data_locations[,1] <- as.numeric(data_i[,1])
  data_locations[,2] <- as.numeric(data_i[,2])
  
  p <- matrix(data=NA,nrow=length(data_i[,3])+dim(boundary)[1],ncol=2)
  p[,1] <- c(data_locations[,1],boundary[,1])
  p[,2] <- c(data_locations[,2],boundary[,2])
  
  isboundary <- matrix(data=NA,nrow=dim(boundary)[1],ncol=2)
  isboundary[,1] <- (length(data_i[,3])+1):(length(data_i[,3])+dim(boundary)[1])
  isboundary[,2] <- c((length(data_i[,3])+2):(length(data_i[,3])+dim(boundary)[1]),(length(data_i[,3])+1))
  
  mesh_1 <- create.MESH.2D(p, order = 1, segments = isboundary)
  mesh <- refine.MESH.2D(mesh_1, maximum_area=0.2, delaunay=TRUE)
  #plot(mesh_1)
  plot(mesh)
  print("iteration n°")
  print(i)
  
  Tri <- mesh$triangles
  basisobj <- create.FEM.basis(mesh)
  kappa <- matrix(0, 2, 2) # Initial value of K
  PDE_parameters <- list(K = kappa, b = c(0, 0), c = 0)
  lambda = 10^seq(3,9,3)
  
  # solve and time
  #t1 <- Sys.time()
  time[i,] <- system.time(
    smoothing_aniso <- aniso.smooth.FEM.PDE.basis(observations = c(data_i[,3],
                                                                   rep(NA,dim(mesh$nodes)[1]-length(data_i[,3]))),
                                                  FEMbasis = basisobj, 
                                                  lambda = rhovec/(1-rhovec)*n/omega, 
                                                  PDE_parameters = PDE_parameters))
  #t2 <- Sys.time()
  #time[i] <- t2-t1
  
  # plot ####
  #plot(smoothing_aniso$fit.FEM)
  
}

plot(data_size, time[,3], type = "b")

