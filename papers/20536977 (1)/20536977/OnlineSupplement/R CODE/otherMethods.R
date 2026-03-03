# source('~/evalFun.R')
library(foreach)
library(doParallel)

# BJS
pca_BJS <- function(Sarray, tall, uall, r, num_cores=10){
  p <- dim(Sarray)[1]; m <- dim(Sarray)[3]
  cores <- num_cores
  cl <- makeCluster(cores)  
  registerDoParallel(cl)
  result <- foreach(i=1:m, .combine = rbind, .export = 'evalFun') %dopar% {
    trueU <- uall[,,i][,1:r]
    truepro <- tcrossprod(trueU, trueU)
    estU <- eigen(Sarray[,,i])$vectors[, 1:r] # pca on sample covariance
    vals <- evalFun(estU, trueU)
    return(vals)
  }
  stopCluster(cl)
  return(result)
}

DT <- function(Sarray, tall, uall, r, n=100, num_cores=10){
  p <- dim(Sarray)[1]; m <- dim(Sarray)[3]
  cores <- num_cores
  cl <- makeCluster(cores)  
  registerDoParallel(cl)
  result <- foreach(i=1:m, .combine = rbind, .export = 'evalFun') %dopar% {
    trueU <- uall[,,i][,1:r]
    # subset & reduced PCA
    vars <- diag(Sarray[,,i])
    sigma2_hat <- quantile(vars, probs = 0.5)
    alpha_n <- 4*sqrt(log(max(n, p))/n)
    delta <- sigma2_hat * (1+alpha_n)
    index_set <- which(vars >= delta)
    estU <- matrix(0, p, r)
    estU[index_set,] <- eigen(Sarray[index_set,index_set,i])$vectors[,1:r]
    # estpro <- tcrossprod(estU, estU)
    # sinangle <- svd(tcrossprod(estpro, diag(rep(1,p))-truepro))$d[1:r]
    # error1 <- norm(sinangle,'2')
    # thresholding
    k <- length(index_set) # number of retained variables
    tau <- apply(estU[index_set,], 2, mad)
    estU <- sapply(1:r, function(j){
      u <- estU[,j]
      u <- ifelse(abs(u)>tau[j], u, 0)
      return(u)
    })
    vals <- evalFun(estU, trueU)
    return(vals)
  }
  stopCluster(cl)
  return(result)
}
