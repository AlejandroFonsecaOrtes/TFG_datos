# source('~/cvband_common.R')
# source('~/ManOpt.R')

library(matrixcalc)
library(Matrix)
library(expm)
library(CVTuningCov)
library(MASS)

# data: n*p*m
# rho: candidate sparsity parameters
# r: dimension of PC's
# k:number of folds, default:5
cvsparsity <- function(data,rho,u,tall,r,cvh,k=5){
  n <- dim(data)[1]; p <- dim(data)[2]; m <- dim(data)[3]
  ntest <- floor(n/k); ntrain <- n - ntest
  cvip <- rep(0,k)
  flag_succ <- rep(0,k)
  for(l in 1:k){
    datatest <- data[((l-1)*ntest+1):(l*ntest),,]
    datatrain <- data[-(((l-1)*ntest+1):(l*ntest)),,]
    out <- evalManPG(datatrain, cvh, u, tall, rho, r)
    temp <- out$U
    flag_succ[l] <- out$flag_succ
    test <- smoothCov(datatest, tall=tall, h=cvh, u)
    cvip[l] <- sum(diag(crossprod(temp, test%*%temp)))
  }
  avcvip = mean(cvip)
  return(list(avcvip = avcvip, cvip = cvip, rho = rho, r = r, h = cvh, flag_succ=flag_succ))
}
