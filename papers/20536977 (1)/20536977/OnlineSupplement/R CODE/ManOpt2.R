library(matrixcalc)
library(Matrix)
library(expm)
library(CVTuningCov)
library(MASS)

# proximal gradient method on manifold optimization
getD <- function(B, U, rho, eta){
  p <- dim(U)[1]; r <- dim(U)[2]
  D <- matrix(0, p, r)
  for(i in 1:p){
    for(j in 1:r){
      if(B[i,j]>eta*rho){
        D[i,j] <- B[i,j]-eta*rho
      }
      else if(B[i,j]<(-eta*rho)){
        D[i,j] <- B[i,j]+eta*rho
      }
      else{
        D[i,j] <- 0
      }
    }
  }
  return(D-U)
}

# update the direction (manifold)
# regularized semi-smooth Newton method
# S: p*p, covariance matrix (local smoothing); A: r*r,Lagrangian multiplier matrix
# U: p*r, eigenvectors; rho: the sparsity parameter
# eta: step-size (proximal)
updateDirection <- function(S, A, U, rho, eta, eta1=0.1, eta2=0.3, lam=0.1, lam0=0.001, max_iter=100){
  p <- dim(U)[1]
  r <- dim(U)[2]; rdim <- r*(r+1)/2
  B <- U + 2*eta*crossprod(S,U) + 2*eta*tcrossprod(U,A)
  D <- getD(B, U, rho=rho, eta=eta)
  E <- crossprod(D, U) + crossprod(U, D)
  vechE <- vech(E); vechA <- vech(A)
  old <- sqrt(F.norm2(E))
  iter <- 0
  while(old^2>1e-10){
    if(r==1){
      temp <- diag(ifelse(abs(B[,1])>rho*eta, 1, 0))
      C <- crossprod(U,temp) %*% U
    }
    else{
      temp <- diag(ifelse(abs(B[,1])>rho*eta, 1, 0))
      C <- crossprod(U,temp) %*% U
      for(i in 2:r){
        temp <- diag(ifelse(abs(B[,i])>rho*eta, 1, 0))
        C <- as.matrix(bdiag(C, crossprod(U,temp) %*% U))
      }
    }
    G <- 4*eta*pinvdupmat %*% C %*% dupmat
    mu <- lam * max(min(old, 0.2), 1e-11) # regularized 
    d <- -solve(G + mu*diag(rep(1, rdim))) %*% vechE
    
    vechA_new <- vechA + d  
    A_new <- matrix(dupmat %*% vechA_new, nrow = r, ncol = r)
    B_new <- U + 2*eta*crossprod(S,U) + 2*eta*tcrossprod(U,A_new)
    D_new <- getD(B_new, U, rho=rho, eta=eta)
    E_new <- crossprod(D_new, U) + crossprod(U, D_new)
    vechE_new <- vech(E_new)
    new <- sqrt(F.norm2(E_new))
    
    rhok <- - sum(E_new * matrix(dupmat%*%d, r, r)) / (norm(d, '2')^2)
    if(new <= 0.99*old){ # newton step
      vechA <- vechA_new; B <- B_new; vechE <- vechE_new
      A <- A_new; D <- D_new
      old <- new
    }else{
      V <- A - sum(E_new*(A-A_new))/new^2*E_new
      B_V <- U + 2*eta*crossprod(S,U) + 2*eta*tcrossprod(U,V)
      D_V <- getD(B_V, U, rho=rho, eta=eta)
      E_V <- crossprod(D_V, U) + crossprod(U, D_V)
      vechE_V <- vech(E_V)
      eps <- sqrt(F.norm2(E_V))
      noiter <- noiter + 1
      if(rhok >= eta1 && eps <= old){ # projection step
        A <- V; vechE <- vechE_V; D <- D_V
        vechA <- vech(A); B <- B_V
        old <- eps
        if(rhok>=eta2){
          lam <- (lam0+lam)/2
        }else{
          lam <- min(2*lam, 10^5)
        }
      }else{
        lam <- min(4*lam, 10^5)
      }
    }
    iter <- iter + 1
    if(iter > max_iter){
      break
    }
  }
  if(iter>max_iter){
    stop_flag <- 1
  }else{
    stop_flag <- 0
  }
  return(list(D=D, A = A, old=old, stop_flag=stop_flag))
}


# retraction: exponential mapping(default)
# alpha : step size
retraction <- function(U, D, alpha=1, type='exp'){
  r <- dim(U)[2]
  switch(type, exp={
    temp <- crossprod(U, D)
    temp1 <- D - U %*% temp
    Q <- qr.Q(qr(temp1))
    R <- qr.R(qr(temp1))
    temp3 <- rbind(temp, R)
    temp4 <- rbind(-t(R), matrix(0, r, r))
    temp5 <- cbind(temp3, temp4)
    temp6 <- as.matrix(expm(alpha*temp5))
    newU <- tcrossprod(crossprod(t(cbind(U, Q)), temp6), 
                       cbind(diag(rep(1,r)), matrix(0, r, r)))
  }, 
  polar={
    D <- alpha * D
    temp <- diag(rep(1,r)) + crossprod(D, D)
    newU <- crossprod(t(U+D), as.matrix(sqrtm(solve(temp))))
  },
  qr={
    qr.U <- qr(U+alpha*D)
    Q <- qr.Q(qr.U)
    R <- qr.R(qr.U)
    newU <- Q %*% diag(diag(sign(R)))
  },
  svd={
    eig <- eigen(crossprod(U+alpha*D))
    newU <- (U+alpha*D) %*% (eig$vectors%*%diag(1/sqrt(eig$values))%*%t(eig$vectors))
  })
  return(newU)
}

getobjValue <- function(S, U, rho){
  temp1 <- -sum(diag(crossprod(U, S) %*% U))
  temp2 <- rho*sum(abs(U))
  return(temp1 + sum(temp2))
}

# Armijo line search procedure
ManPG <- function(S, U, D, rho, delta=0.001, alpha=1, gamma=0.5, type='exp'){
  old <- getobjValue(S, U, rho=rho)
  newU <- retraction(U, D, alpha = alpha, type = type)
  new <- getobjValue(S, newU, rho=rho)
  while(new > old - delta*alpha*F.norm2(D)){
    alpha <- gamma * alpha
    newU <- retraction(U, D, alpha = alpha, type = type)
    new <- getobjValue(S, newU, rho=rho)
  }
  return(newU)
}

# rho:sparsity parameter; r: dimension of PC's
# tall: common-vector; irregular-list.
evalManPG <- function(data, h, u, tall, rho, r, type='exp', tol=1e-8, max_iter=1000){
  n <- dim(data)[1]; p <- dim(data)[2]
  S <- smoothCov(data, tall, h=h, u)
  eta <- 1 / (2*max(eigen(S)$values)) # step-size (proximal)
  U <- eigen(S)$vectors[,1:r,drop=F]
  # Lagrangian multiplier matrix A
  A <- matrix(0, r, r)
  oldobj <- getobjValue(S, U, rho=rho)
  out <- updateDirection(S, A, U, rho=rho, eta=eta)
  D <- out[[1]]
  # A <- out$A
  newU <- ManPG(S, U, D, rho = rho, type = type)
  newobj <- getobjValue(S, newU, rho=rho)
  iter <- 0
  while(F.norm2(D)>tol){
    out <- updateDirection(S, A, newU, rho=rho, eta=eta)
    D <- out[[1]]
    # A <- out$A
    newU <- ManPG(S, newU, D, rho = rho, type = type)
    oldobj <- newobj
    newobj <- getobjValue(S, newU, rho=rho)
    iter <- iter + 1
    if(iter>max_iter){
      break
    }
  }
  if(iter>max_iter && sqrt(F.norm2(D))/eta > 0.1){
    print('unsuccessful!')
    flag_succ <- 0
  }else{
    flag_succ <- 1
  }
  return(list(U=newU, obj=newobj, smoothS=S, D=D, h=h, evalt=u, rho=rho, 
              ftol1=F.norm2(D), ftol2=abs(oldobj - newobj), flag_succ=flag_succ)) 
}
