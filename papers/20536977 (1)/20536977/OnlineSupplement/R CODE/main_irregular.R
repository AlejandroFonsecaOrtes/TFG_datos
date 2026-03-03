source('~/dataGe.R')
source('~/cvband_irregular.R')
source('~/cvsparsity_irregular.R')
source('~/ManOpt.R')
source('~/dcm_cvband_irregular.R')
source('~/dcm_cvthrs_irregular.R')
source('~/evalFun.R')

library(matrixcalc)
library(Matrix)
library(expm)
library(CVTuningCov)
library(MASS)
library(foreach)
library(doParallel)
library(optparse)

opts_list <- list(make_option('--n', type='integer', default=500, help='sample size'),
                  make_option('--p', type='integer', default=200, help='dimension'),
                  make_option('--m', type='character', default='19,20,21', help='number of grids'),
                  make_option('--noise', type='double', default=3, help='variance of measurement error'))
opts_args <- OptionParser(option_list = opts_list)
opts <- parse_args(opts_args)

n <- opts$n; p <- opts$p
mseq <- as.numeric(strsplit(opts$m, ',')[[1]])
print(mseq)
m <- mean(mseq)
noise <- opts$noise
type <- 'irregular'
num_cores <- 10

# r: dimension of principal subspace
r <- 3; rdim <- r*(r+1)/2
if(r>1){
  dupmat <- duplication.matrix(r)
  pinvdupmat <- tcrossprod(solve(crossprod(dupmat)), dupmat)
}else{
  dupmat <- matrix(1,1,1)
  pinvdupmat <- matrix(1,1,1)
}
evalu <- 2*(1:100)/(2*100+1)
evalu <- evalu[seq(2,100,length.out = 50)]
evalulen <- length(evalu)
interval <- evalu[2]-evalu[1]
uall <- getTrue(evalu, p)

# candidate parameters
hh <- seq(log(0.015),log(0.5),length.out = 10)
hlist <- exp(hh)[1:8]
dcm_hlist <- hlist
logrho<-seq(log(0.5), log(2), length.out = 6)
rholist<-exp(logrho)
rholen <- length(rholist)
qlist <- c(13, 14, 15, 16, 18)
qlen <- length(qlist)

nrun <- 100
run <- 1
while(run <= nrun){
  print(paste0('Run: ', run))
  
  temp <- dataGe(n, p, mseq, type, sigma2_noise = noise)
  data <- temp$data; tall <- temp$tall
  
  # tune the bandwidth
  print(paste0('tune the bandwidth, run: ', run))
  cvhout <- cvband(data, tall, hlist, r=r, num_cores=num_cores)
  cvh <- cvhout$cvh

  # tune the sparsity level
  cv_result <- cv_flag_succ <- matrix(0, rholen, evalulen)
  for(j in 1:rholen){
    rho <- rholist[j]
    print(paste0('tune the sparsity level, Run: ', run, 'sparsity: ', j))
    cores <- num_cores
    cl <- makeCluster(cores)  
    registerDoParallel(cl)
    result <- foreach(i=1:evalulen, .packages = c('CVTuningCov', 'matrixcalc', 'Matrix', 'expm', 'MASS', 'fda', 'glmnet')) %dopar% {
      u <- evalu[i]
      cvout <- cvsparsity(data, rho, u, tall, r, cvh = cvh)
      flag_succ <- cvout$flag_succ
      cvip <- cvout$avcvip
      return(c(cvip, flag_succ))
    }
    stopCluster(cl)
    for(i in 1:evalulen){
      cv_result[j,i] <- result[[i]][1]
      cv_flag_succ[j,i] <- all(result[[i]][-1]==1)
    }
  }
  if(any(cv_flag_succ==0)){
    next
  }
  cvrhoseq <- rholist[apply(cv_result, 2, which.max)]

  # refine
  print(paste0('tune the threshold, Run: ', run))
  cores <- num_cores
  cl <- makeCluster(cores)  
  registerDoParallel(cl)
  cvthrsseq <- foreach(i=1:evalulen, .combine = rbind, .packages = c('CVTuningCov', 'matrixcalc', 'Matrix', 'expm', 'MASS', 'fda', 'glmnet')) %dopar% {
    u <- evalu[i]
    out <- evalManPG(data, cvh, u, tall, cvrhoseq[i], r)
    estU <- out$U 
    if(out$flag_succ==0){
      return(-1)
    }
    estpro <- tcrossprod(estU, estU)
    diag_element <- sort(diag(estpro), decreasing = T)
    thrs_cvip <- thrs_cv_flag_succ <- rep(0, qlen)
    for(j in 1:qlen){
      q <- qlist[j]
      thrs <- diag_element[q]
      ind <- which(diag(estpro)>=thrs)
      cvout <- cvsparsity(data[,ind,,drop=FALSE], cvrhoseq[i], u, tall, r, cvh = cvh)
      thrs_cv_flag_succ[j] <- all(cvout$flag_succ==1)
      thrs_cvip[j] <- cvout$avcvip
    }
    index_max <- which.max(thrs_cvip)
    cvthrs <- qlist[index_max]
    if(any(thrs_cv_flag_succ==0)){
      cvthrs <- -1
    }
    return(cvthrs)
  }
  stopCluster(cl)
  if(any(cvthrsseq==-1)){
    next
  }

  print(paste0('evaluation, Run:', run))
  cores <- num_cores
  cl <- makeCluster(cores)  
  registerDoParallel(cl)
  result_ours <- foreach(i=1:evalulen, .combine = rbind, .packages = c('CVTuningCov', 'matrixcalc', 'Matrix', 'expm', 'MASS', 'fda', 'glmnet')) %dopar% {
    u <- evalu[i]
    out <- evalManPG(data, cvh, u, tall, cvrhoseq[i], r)
    estU <- out$U
    if(out$flag_succ==0){
      return(rep(-1,6))
    }
    trueU <- uall[,,i][,1:r]
    vals <- evalFun(estU, trueU)
    
    # thresholding 
    estpro <- tcrossprod(estU, estU)
    diag_element <- sort(diag(estpro), decreasing = T)
    q <- cvthrsseq[i]
    thrs <- diag_element[q]
    ind <- intersect(which(diag(estpro)>=thrs), which(diag(estpro)>1e-6))
    # refined estimate
    refU <- matrix(0, p, r)
    out <- evalManPG(data[, ind, ,drop=FALSE], cvh, u, tall, cvrhoseq[i], r)
    refU[ind, ] <- out$U
    if(out$flag_succ==0){
      return(rep(-1,6))
    }
    thrs_vals <- evalFun(refU, trueU)
    return(c(vals, thrs_vals))
  }
  stopCluster(cl)
  if(any(result_ours==-1)){
    next
  }

  ############# dcm+ ###############
  # tune the bandwidth
  print(paste0('dcmr, tune the bandwidth, run: ', run))
  cvhout <- dcmr_cvband(data, tall, dcm_hlist)
  cvh <- cvhout$cvh
  
  # tune thresholding parameter
  print(paste0('dcmr, tune the thresholding parameter, Run: ', run))
  cores <- num_cores
  cl <- makeCluster(cores)  
  registerDoParallel(cl)
  dcmr_thresh <- foreach(j=1:evalulen, .combine = rbind, .packages = c('CVTuningCov', 'matrixcalc', 'Matrix', 'expm', 'MASS', 'fda', 'glmnet')) %dopar% {
    u <- evalu[j]
    cvout <- cvthreshold(data, tall, cvh, u)
    thrs <- cvout$threshold
    return(thrs)
  }
  stopCluster(cl)

  ### eigen-decomposition ###
  print(paste0('dcmr, evaluation, Run:', run))
  cores <- num_cores
  cl <- makeCluster(cores)  
  registerDoParallel(cl)
  result_dcmr <- foreach(i=1:evalulen, .combine = rbind, .packages = c('CVTuningCov', 'matrixcalc', 'Matrix', 'expm', 'MASS', 'fda', 'glmnet')) %dopar% {
    u <- evalu[i]
    thresh <- dcmr_thresh[i]
    estcov <- dcm_smoothCov(data, tall, cvh, u)
    estcov <- sign(estcov) * (abs(estcov) - thresh) * ((abs(estcov) - thresh) > 0)
    estU <- eigen(estcov)$vectors[,1:r]
    trueU <- uall[,,i][,1:r]
    vals <- evalFun(estU, trueU)
    return(vals)
  }
  stopCluster(cl)
  
  ############# dcm ###############
  # tune the bandwidth
  print(paste0('dcm, tune the bandwidth, run: ', run))
  cvhout <- dcm_cvband(data, tall, dcm_hlist)
  cvh <- cvhout$cvh

  # tune thresholding parameter
  print(paste0('dcm, tune the thresholding parameter, Run: ', run))
  cores <- num_cores
  cl <- makeCluster(cores)  
  registerDoParallel(cl)
  dcm_thresh <- foreach(j=1:evalulen, .combine = rbind, .packages = c('CVTuningCov', 'matrixcalc', 'Matrix', 'expm', 'MASS', 'fda', 'glmnet')) %dopar% {
    u <- evalu[j]
    cvout <- cvthreshold(data, tall, cvh, u)
    thrs <- cvout$threshold
    return(thrs)
  }
  stopCluster(cl)

  ### eigen-decomposition ###
  print(paste0('dcm, evaluation, Run:', run))
  cores <- num_cores
  cl <- makeCluster(cores)  
  registerDoParallel(cl)
  result_dcm <- foreach(i=1:evalulen, .combine = rbind, .packages = c('CVTuningCov', 'matrixcalc', 'Matrix', 'expm', 'MASS', 'fda', 'glmnet')) %dopar% {
    u <- evalu[i]
    thresh <- dcm_thresh[i]
    estcov <- dcm_smoothCov(data, tall, cvh, u)
    estcov <- sign(estcov) * (abs(estcov) - thresh) * ((abs(estcov) - thresh) > 0)
    estU <- eigen(estcov)$vectors[,1:r]
    trueU <- uall[,,i][,1:r]
    vals <- evalFun(estU, trueU)
    return(vals)
  }
  stopCluster(cl)
  # save the results if necessary
  run <- run + 1
}
