library(matrixcalc)
library(Matrix)
library(CVTuningCov)
library(MASS)
library(foreach)
library(doParallel)

Ekernel <- function(x) {ifelse(abs(x)<=1, 3*(1-x^2)/4, 0)}

# Sarray: array of sample covariance matrices (p*p*m)
dcm_smoothCov <- function(data, tall, h, t_eval){
  n <- dim(data)[1]; p <- dim(data)[2]; m <- dim(data)[3]
  Sarray <- array(0, dim = c(p,p,m))
  for(j in 1:m){
    Sarray[,,j] <- cov(data[,,j])
  }
  if(h==0 && t_eval%in%tall){  
    return(Sarray[,,which(t_eval==tall)])
  }else if(!any(abs(tall-t_eval)<=h)){
    print('No observational points within the bandwidth!')
    return(0)
    break
  }
  else{
    p <- dim(Sarray)[1]
    tind <- which(abs(tall-t_eval)<=h)
    l <- length(tind)
    ttemp <- tall[tind]
    tempS <- Sarray[,,tind,drop=F]
    temp <- matrix(0, p, p)
    for(i in 1:l){
      temp <- Ekernel((ttemp[i] - t_eval)/h) * tempS[,,i] + temp
    }
    temp1 <- sum(Ekernel((ttemp-t_eval)/h))
    return(temp/temp1)
  }
}

# tune the bandwidth given rho=o
# data: n*p*m
# hlist: candidate bandwidth parameters
# leave-one-curve-out
dcmr_cvband <- function(data,tall,hlist,rho=0, num_cores=10){
  n <- dim(data)[1]; p <- dim(data)[2]; m <- dim(data)[3]
  hlen <- length(hlist)
  NN = 20 # number of repetitions
  pp = min(round(p/12), 8) # subset of variables for cross-validation
  varset = matrix(0, nrow = NN, ncol = pp)
  for (j in 1:NN) {
    a = sample(1:p, pp, replace = FALSE, prob = NULL)
    a = sort(a)
    varset[j, ] = a
  }
  mm <- min(m, 10)
  cores <- num_cores
  cl <- makeCluster(cores)
  registerDoParallel(cl)
  result <- foreach(l = 1:n, .combine=rbind, .export = c('Ekernel', 'dcm_smoothCov')) %dopar% {
    datatest <- data[l,,]
    datatrain <- data[-l,,]
    # randomly sample mm time points of each curve for validation
    ind_select <- sample(1:m, mm, replace = F)
    t_select <- tall[ind_select]
    error <- rep(0, hlen)
    for(j in 1:hlen){
      h <- hlist[j]
      for(jj in 1:NN){
        dim_idx = varset[jj,]
        for(i in 1:mm){
          u <- t_select[i]
          S <- dcm_smoothCov(datatrain[,dim_idx,], tall=tall, h=h, t_eval=u)
          # rho=0, standard PCA
          error[j] <- error[j] + t(datatest[dim_idx, ind_select[i]]) %*% solve(S) %*% t(t(datatest[dim_idx, ind_select[i]])) + log(det(S))
        }
      }
    }
    error <- error/mm/NN
    return(error)
  }
  stopCluster(cl)
  error <- colMeans(result)
  return(list(cvh = hlist[which.min(error)], error = error, hlist = hlist))
}

# leave one point out
dcm_cvband <- function(data, tall, hlist, num_cores=10) {
  
  n = dim(data)[1]; m = dim(data)[3]
  p = dim(data)[2]
  Sarray <- array(0, dim = c(p,p,m))
  for(j in 1:m){
    Sarray[,,j] <- cov(data[,,j])
  }
  hlen = length(hlist)
  
  ######################################
  #### calculate the bandwidth parameter
  ######################################
  
  NN = 20 # number of repetitions
  pp = min(round(p/12),8) ## subset of variables for cross-validation
  varset = matrix(0, nrow = NN, ncol = pp)
  for (j in 1:NN) {
    a = sample(1:p, pp, replace = FALSE, prob = NULL)
    a = sort(a)
    varset[j, ] = a
  }
  mm = min(m, 10)
  cores <- num_cores
  cl <- makeCluster(cores)
  registerDoParallel(cl)
  result <- foreach(shu = 1:n, .combine=rbind, .export = c('Ekernel', 'dcm_smoothCov')) %dopar% {
    # randomly sample mm time points of each curve for validation
    ind_val_select <- sample(1:m, mm, replace = F)
    tval_select <- tall[ind_val_select]
    error.band = rep(0, hlen)
    for(band in 1:hlen){
      h = hlist[band]
      sum.NW = 0
      for (j in 1:NN) {
        dim_idx = varset[j,] # sample a subset of variables
        for(ii in 1:mm){
          ind_val = ind_val_select[ii]
          u = tval_select[ii]
          sigma.NW = matrix(0, pp, pp)
          weight.NW = 0
          K = Ekernel((tall - u)/h)/h
          tind <- which(abs(tall-u)<=h)
          l <- length(tind)
          for(ti in 1:l){
            i = tind[ti]
            if(i != ind_val){
              sigma.NW = sigma.NW + K[i] * Sarray[dim_idx,dim_idx,i]*(n-1)
              weight.NW = weight.NW + K[i]*n
            }
            else{
              data_train = data[-shu,dim_idx,i]
              data_test = data[shu, dim_idx, i]
              sigma.NW = sigma.NW + K[i] * cov(data_train)*(n-2)
              weight.NW = weight.NW + K[i]*(n-1)
            }
          }
          sigma.NW = sigma.NW / weight.NW
          sum.NW = sum.NW + t(data_test) %*% solve(sigma.NW) %*% t(t(data_test)) + log(det(sigma.NW))
        }
      }
      error.band[band] = sum.NW/NN/mm
    }
    return(error.band)
  }
  stopCluster(cl)
  cvh = hlist[which.min(colMeans(result))]
  return(list(cvh=cvh, error=colMeans(result), hlist=hlist))
}
