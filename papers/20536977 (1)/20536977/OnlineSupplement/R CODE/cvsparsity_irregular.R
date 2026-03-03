library(matrixcalc)
library(Matrix)
library(expm)
library(CVTuningCov)
library(MASS)

# source('~/cvband_irregular.R')
# source('~/ManOpt.R')

# tune the sparsity parameter given the bandwidth
# data: n*p*m
# rho: candidate sparsity parameters
# r: dimension of PC's
# k:number of folds, default:5
cvsparsity <- function(data,rho,u,tall,r,cvh,k=5){
  n <- dim(data)[1]; p <- dim(data)[2]
  ntest <- floor(n/k); ntrain <- n - ntest
  cvip <- flag_succ <- rep(0,k)
  for(l in 1:k){
    datatest <- data[((l-1)*ntest+1):(l*ntest),,]
    datatrain <- data[-(((l-1)*ntest+1):(l*ntest)),,]
    talltest <- tall[((l-1)*ntest+1):(l*ntest)]
    talltrain <- tall[-(((l-1)*ntest+1):(l*ntest))]
    out <- evalManPG(datatrain, cvh, u, talltrain, rho, r)
    U <- out$U
    flag_succ[l] <- out$flag_succ
    cov_test <- smoothCov(datatest, tall=talltest, h=cvh, u)
    cvip[l] <- sum(diag(crossprod(U, cov_test%*%U)))
  }
  return(list(avcvip = mean(cvip), cvip = cvip, rho = rho, r = r, h = cvh, flag_succ=flag_succ))
}
