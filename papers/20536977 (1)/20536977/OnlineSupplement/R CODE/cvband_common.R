library(matrixcalc)
library(foreach)
library(doParallel)

# local linear smoothing
K <- function(x) {ifelse(abs(x)<=1, 3*(1-x^2)/4, 0)}

# weights for the local linear smoothing
weight_ele <- function(u, x, h){
  s0 <- sum(K((u-x)/h))
  s1 <- sum(K((u-x)/h)*(u-x))
  s2 <- sum(K((u-x)/h)*(u-x)^2)
  return(c(s0,s1,s2))
}

# Sarray: array of sample covariance matrices (p*p*m)
smoothCov <- function(data, tall, h, t_eval){
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
    ss <- weight_ele(tall, t_eval, h)
    tind <- which(abs(tall-t_eval)<=h)
    l <- length(tind)
    ttemp <- tall[tind]
    tempS <- Sarray[,,tind,drop=F]
    temp <- matrix(0, p, p)
    for(i in 1:l){
      temp <- (ss[3]*K((ttemp[i] - t_eval)/h)-ss[2]*K((ttemp[i] - t_eval)/h)*(ttemp[i]-t_eval)) * tempS[,,i] + temp
    }
    temp1 <- ss[1]*ss[3]-ss[2]^2
    if(temp1==0){
      temp1<- temp1+1/n^2
    }
    return(temp/temp1)
  }
}

# function to tune bandwidth given rho=o
# data: n*p*m
# hlist: candidate bandwidth parameters
# r: dimension of PC's
# leave-one-curve-out
cvband <- function(data,tall,hlist,r,rho=0, num_cores=10){
  n <- dim(data)[1]; p <- dim(data)[2]; m <- dim(data)[3]
  hlen <- length(hlist)
  mm <- min(m, 10)
  cores <- num_cores
  cl <- makeCluster(cores)
  registerDoParallel(cl)
  result <- foreach(l = 1:n, .combine=rbind, .export = c('K', 'weight_ele', 'smoothCov')) %dopar% {
    datatest <- data[l,,]
    datatrain <- data[-l,,]
    ind_select <- sample(1:m, mm, replace = F)
    t_select <- tall[ind_select]
    cvip <- rep(0, hlen)
    for(j in 1:hlen){
      h <- hlist[j]
      for(i in 1:mm){
        u <- t_select[i]
        ind <- ind_select[i]
        S <- smoothCov(datatrain, tall=tall, h=h, u)
        # rho=0, standard PCA
        temp <- eigen(S)$vectors[,1:r,drop=F]
        cvip[j] <- cvip[j] + sum(crossprod(temp, datatest[,ind,drop=F])^2)
      }
    }
    return(cvip)
  }
  stopCluster(cl)
  cvip <- colSums(result)/(mm*n)
  return(list(cvh = hlist[which.max(cvip)], cvip = cvip, error=-cvip, hlist = hlist, r = r))
}