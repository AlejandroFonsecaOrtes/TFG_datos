library(matrixcalc)
library(Matrix)
library(CVTuningCov)
library(MASS)
library(foreach)
library(doParallel)

Ekernel <- function(x) {ifelse(abs(x)<=1, 3*(1-x^2)/4, 0)}

dcm_smoothMean <- function(data, tall, h, u){
  n <- dim(data)[1]; p <- dim(data)[2]
  temp <- rep(0, p)
  temp1 <- 0
  for(i in 1:n){
    tall2 <- tall[[i]]
    if(all(abs(tall2-u)>h)){
      next
    }else{
      tind <- which(abs(tall2-u)<=h)
      l <- length(tind)
      ttemp <- tall2[tind]
      for(j in 1:l){
        temp <- Ekernel((ttemp[j] - u)/h) * data[i,,tind[j]] + temp
        temp1 <- Ekernel((ttemp[j] - u)/h) + temp1
      }
    }
  }
  return(temp/temp1)
}

dcm_smoothCov <- function(data, tall, h, u){
  n <- dim(data)[1]
  p <- dim(data)[2]
  temp <- matrix(0, p, p)
  temp1 <- 0
  for(i in 1:n){
    data1 <- data[i,,]
    tall2 <- tall[[i]]
    if(all(abs(tall2-u)>h)){
      next
    }else{
      tind <- which(abs(tall2-u)<=h)
      l <- length(tind)
      ttemp <- tall2[tind]
      for(j in 1:l){
        temp <- Ekernel((ttemp[j] - u)/h) * tcrossprod(data1[,tind[j],drop=F]) + temp
        temp1 <- Ekernel((ttemp[j] - u)/h) + temp1
      }
    }
  }
  smooth_mean <- matrix(dcm_smoothMean(data, tall, h, u), p, 1)
  return(temp/temp1 - tcrossprod(smooth_mean))
}

dcmr_cvband <- function(data,tall,hlist,rho=0, num_cores=10){
  n <- dim(data)[1]; p <- dim(data)[2]
  hlen <- length(hlist)
  
  NN = 20 # number of repetitions
  pp = min(round(p/12), 5) # subset of variables for cross-validation
  varset = matrix(0, nrow = NN, ncol = pp)
  for (j in 1:NN) {
    a = sample(1:p, pp, replace = FALSE, prob = NULL)
    a = sort(a)
    varset[j, ] = a
  }
  cores <- num_cores
  cl <- makeCluster(cores)
  registerDoParallel(cl)
  result <- foreach(l = 1:n, .combine=rbind, .export = c('Ekernel', 'dcm_smoothMean', 'dcm_smoothCov')) %dopar% {
    datatest <- data[l,,]
    datatrain <- data[-l,,]
    talltrain <- tall[-l]
    ts <- tall[[l]]
    m <- min(length(ts), 10)
    t_ind <- sample(length(ts), m, replace = F)
    error <- rep(0, hlen)
    for(i in 1:hlen){
      h <- hlist[i]
      for(jj in 1:NN){
        a = varset[jj,]
        for(j in 1:m){
          evalt_ind <- t_ind[j]
          u<-ts[evalt_ind]
          S <- dcm_smoothCov(datatrain[,a,], talltrain, h=h, u)
          error[i] <- error[i] + t(datatest[a, evalt_ind]) %*% solve(S) %*% t(t(datatest[a, evalt_ind])) + log(det(S))
        }
      }
    }
    return(error)
  }
  stopCluster(cl)
  count <- sapply(tall, length)
  count <- ifelse(count>10, 10, count)
  error <- colSums(result)/sum(count)/NN
  return(list(cvh = hlist[which.min(error)], error = error, hlist = hlist))
}

dcm_cvband <- function(data, tall, hlist, num_cores=10) {
  nn = dim(data)[1]
  Y <- NULL
  X <- c()
  for(i in 1:nn){
    m <- length(tall[[i]])
    Y <- rbind(Y, t(data[i,,1:m]))
    X <- c(X, tall[[i]])
  }
  count <- sapply(tall, length)
  n  = length(X)
  p = length(Y[1, ])
  hlen = length(hlist)

  ######################################
  #### calculate the bandwidth parameter
  ######################################
  
  NN = 20 # number of repetitions
  pp = min(round(p/12),5) # subset of variables for cross-validation
  varset = matrix(0, nrow = NN, ncol = pp)
  for (j in 1:NN) {
    a = sample(1:p, pp, replace = FALSE, prob = NULL)
    a = sort(a)
    varset[j, ] = a
  }
  cores <- num_cores
  cl <- makeCluster(cores)
  registerDoParallel(cl)
  result <- foreach(shu = 1:nn, .combine=rbind, .export = c('Ekernel', 'dcm_smoothMean', 'dcm_smoothCov')) %dopar% {
    talltest <- tall[[shu]]
    m <- count[shu]
    mm <- min(m, 10)
    ind_val_select <- sort(sample(1:m, mm, replace = F))
    tval_select <- talltest[ind_val_select]
    error.band = numeric(hlen)
    for(band in 1:hlen){
      h = hlist[band]
      sum.NW = 0
      for(ti in 1:mm){
        x <- tval_select[ti]
        ind_val = ind_val_select[ti]
        ind_val2 = max(cumsum(count[1:(shu-1)]))+ind_val
        for(j in 1:NN){
          a = varset[j,]
          YY = Y[,a]
          K = Ekernel((X - x)/h)/h
          index = which(abs(X-x)<h)
          l = length(index)
          sigma.NW.jihe = matrix(0, ncol = pp, nrow = pp)
          for (ii in 1:l) {
            i = index[ii]
            if (i != ind_val2) {
              sigma.NW.jihe = sigma.NW.jihe + K[i] * (t(t(YY[i, ])) %*% t(YY[i, ]))
            }
          }
          sigma.NW.jihe = sigma.NW.jihe/sum(K[-ind_val2])
          sum.NW = sum.NW + t(YY[ind_val2, ]) %*% solve(sigma.NW.jihe) %*% t(t(YY[ind_val2, ])) + log(det(sigma.NW.jihe))
        }
      }
      error.band[band] <- sum.NW
    }
    return(error.band)
  }
  stopCluster(cl)
  count <- ifelse(count>10, 10, count)
  error <- colSums(result)/sum(count)/NN
  cvh = hlist[which.min(error)]
  return(list(cvh=cvh, error=error, hlist=hlist))
}
