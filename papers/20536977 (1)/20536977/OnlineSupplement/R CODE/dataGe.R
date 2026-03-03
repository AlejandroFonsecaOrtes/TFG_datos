# function files to generate simulated data
library(fda)
library(MASS)

# m: the number of time points; common: scalar; irregular: sequence
# sigma2_noise: variance of random noise
dataGe <- function(n, p, m, type='common', sigma2_noise=1){
  # eigenvalues
  lambda <- c(30, 18, 10, 5, 3, 2, 1, 0.5, 0.2, 0.1)
  lammat <- diag(lambda)
  
  if(type=='common'){
    tall <- 2*(1:m)/(2*m+1)
    # true eigenvectors
    indmat <- matrix(1:50, nrow = 5, ncol = 10)
    fourier <- create.fourier.basis(c(0,1), nbasis = 7, dropind = 1)
    basismat <- eval.basis(fourier, tall)[,1:5] # row:m
    uall <- array(0, dim = c(p,10,m))
    for(i in 1:m){
      temp <- NULL
      for(j in 1:10){
        ind <- indmat[,j]
        tempu <- rep(0, p)
        tempu[ind] <- basismat[i,]
        temp <- cbind(temp, tempu)
      }
      uall[,,i] <- qr.Q(qr(temp))
    }
    # generate data
    data <- array(0, dim = c(n,p,m))
    score <- mvrnorm(n, mu=rep(0,10), Sigma = lammat) # n*p
    for(i in 1:m){
      noise <- mvrnorm(n, mu=rep(0,p), Sigma = diag(rep(sigma2_noise, p)))
      data[,,i] <- tcrossprod(score, uall[,,i]) + noise
    }
  }
  if(type=='irregular'){
    
    tall <- list()
    m_max <- 1
    for(i in 1:n){
      mm <- sample(m, 1, replace = FALSE)
      tall[[i]] <- runif(mm)
      m_max <- max(m_max, mm)
    }

    # Fourier basis
    indmat <- matrix(1:50, nrow = 5, ncol = 10)
    fourier <- create.fourier.basis(c(0,1), nbasis = 7, dropind = 1)
    
    data <- array(0, dim = c(n,p,m_max))
    score <- mvrnorm(n, mu=rep(0,10), Sigma = lammat) # n*p
    for(i in 1:n){
      t <- tall[[i]]; m <- length(t)
      noise <- mvrnorm(m, mu=rep(0,p), Sigma = diag(rep(sigma2_noise, p)))
      basismat <- eval.basis(fourier, t)[,1:5] # row:m
      uall <- array(0, dim = c(p,10,m))
      for(ii in 1:m){
        temp <- NULL
        for(jj in 1:10){
          ind <- indmat[,jj]
          tempu <- rep(0, p)
          tempu[ind] <- basismat[ii,]
          temp <- cbind(temp, tempu)
        }
        # principal eigenvectors
        uall[,,ii] <- qr.Q(qr(temp))
        data[i,,ii] <- uall[,,ii] %*% matrix(score[i,], 10, 1) + noise[ii,]
      }
    }
  }
  return(list(data=data, tall=tall))
}

getTrue <- function(t, p){
  m <- length(t)
  # true eigenvectors and eigenvalues
  indmat <- matrix(1:50, nrow = 5, ncol = 10)
  fourier <- create.fourier.basis(c(0,1), nbasis = 7, dropind = 1)
  basismat <- eval.basis(fourier, t)[,1:5]
  basismat <- matrix(basismat, ncol = 5)
  uall <- array(0, dim = c(p,10,m))
  for(i in 1:m){
    temp <- NULL
    for(j in 1:10){
      ind <- indmat[,j]
      tempu <- rep(0, p)
      tempu[ind] <- basismat[i,]
      temp <- cbind(temp, tempu)
    }
    uall[,,i] <- qr.Q(qr(temp))
  }
  return(uall)
}
