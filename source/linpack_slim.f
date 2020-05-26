C
C    this file contains all linpack and blas routines required by GMA
C
      SUBROUTINE DPODI(A,N,DET,JOB)                                 
      INTEGER N,JOB                                                 
      DOUBLE PRECISION, INTENT(INOUT) :: A(N,N)
      DOUBLE PRECISION DET(2)                                           
      DOUBLE PRECISION T                                                
      DOUBLE PRECISION S                                                
      INTEGER I,J,JM1,K,KP1                                             
      IF (JOB/10 .EQ. 0) GO TO 70                                       
         DET(1) = 1.0D0                                                 
         DET(2) = 0.0D0                                                 
         S = 10.0D0                                                     
         DO 50 I = 1, N                                                 
            DET(1) = A(I,I)**2*DET(1)                                   
            IF (DET(1) .EQ. 0.0D0) GO TO 60                             
   10       IF (DET(1) .GE. 1.0D0) GO TO 20                             
               DET(1) = S*DET(1)                                        
               DET(2) = DET(2) - 1.0D0                                  
            GO TO 10                                                    
   20       CONTINUE                                                    
   30       IF (DET(1) .LT. S) GO TO 40                                 
               DET(1) = DET(1)/S                                        
               DET(2) = DET(2) + 1.0D0                                  
            GO TO 30                                                    
   40       CONTINUE                                                    
   50    CONTINUE                                                       
   60    CONTINUE                                                       
   70 CONTINUE                                                          
      IF (MOD(JOB,10) .EQ. 0) GO TO 140                                 
         DO 100 K = 1, N                                                
            A(K,K) = 1.0D0/A(K,K)                                       
            T = -A(K,K)                                                 
            CALL DSCAL(K-1,T,A(1,K),1)                                  
            KP1 = K + 1                                                 
            IF (N .LT. KP1) GO TO 90                                    
            DO 80 J = KP1, N                                            
               T = A(K,J)                                               
               A(K,J) = 0.0D0                                           
               CALL DAXPY(K,T,A(1,K),1,A(1,J),1)                        
   80       CONTINUE                                                    
   90       CONTINUE                                                    
  100    CONTINUE                                                       
         DO 130 J = 1, N                                                
            JM1 = J - 1                                                 
            IF (JM1 .LT. 1) GO TO 120                                   
            DO 110 K = 1, JM1                                           
               T = A(K,J)                                               
               CALL DAXPY(K,T,A(1,J),1,A(1,K),1)                        
  110       CONTINUE                                                    
  120       CONTINUE                                                    
            T = A(J,J)                                                  
            CALL DSCAL(J,T,A(1,J),1)                                    
  130    CONTINUE                                                       
  140 CONTINUE                                                          
      RETURN                                                            
      END                                                               


      SUBROUTINE DPOFA(A,N,INFO)                                    
      INTEGER N,INFO                                                
      DOUBLE PRECISION, INTENT(INOUT) :: A(N,N)
      DOUBLE PRECISION DDOT,T                                           
      DOUBLE PRECISION S                                                
      INTEGER J,JM1,K                                                   
         DO 30 J = 1, N                                                 
            INFO = J                                                    
            S = 0.0D0                                                   
            JM1 = J - 1                                                 
            IF (JM1 .LT. 1) GO TO 20                                    
            DO 10 K = 1, JM1                                            
               T = A(K,J) - DDOT(K-1,A(1,K),1,A(1,J),1)                 
               T = T/A(K,K)                                             
               A(K,J) = T                                               
               S = S + T*T                                              
   10       CONTINUE                                                    
   20       CONTINUE                                                    
            S = A(J,J) - S                                              
            IF (S .LE. 0.0D0) GO TO 40                                  
            A(J,J) = DSQRT(S)                                           
   30    CONTINUE                                                       
         INFO = 0                                                       
   40 CONTINUE                                                          
      RETURN                                                            
      END                                                               


      SUBROUTINE DPPDI(AP,N,DET,JOB)                                    
      INTEGER N,JOB                                                     
      DOUBLE PRECISION, INTENT(INOUT) :: AP(N*(N+1)/2)
      DOUBLE PRECISION DET(2)                                           
      DOUBLE PRECISION T                                                
      DOUBLE PRECISION S                                                
      INTEGER I,II,J,JJ,JM1,J1,K,KJ,KK,KP1,K1                           
      IF (JOB/10 .EQ. 0) GO TO 70                                       
         DET(1) = 1.0D0                                                 
         DET(2) = 0.0D0                                                 
         S = 10.0D0                                                     
         II = 0                                                         
         DO 50 I = 1, N                                                 
            II = II + I                                                 
            DET(1) = AP(II)**2*DET(1)                                   
            IF (DET(1) .EQ. 0.0D0) GO TO 60                             
   10       IF (DET(1) .GE. 1.0D0) GO TO 20                             
               DET(1) = S*DET(1)                                        
               DET(2) = DET(2) - 1.0D0                                  
            GO TO 10                                                    
   20       CONTINUE                                                    
   30       IF (DET(1) .LT. S) GO TO 40                                 
               DET(1) = DET(1)/S                                        
               DET(2) = DET(2) + 1.0D0                                  
            GO TO 30                                                    
   40       CONTINUE                                                    
   50    CONTINUE                                                       
   60    CONTINUE                                                       
   70 CONTINUE                                                          
      IF (MOD(JOB,10) .EQ. 0) GO TO 140                                 
         KK = 0                                                         
         DO 100 K = 1, N                                                
            K1 = KK + 1                                                 
            KK = KK + K                                                 
            AP(KK) = 1.0D0/AP(KK)                                       
            T = -AP(KK)                                                 
            CALL DSCAL(K-1,T,AP(K1),1)                                  
            KP1 = K + 1                                                 
            J1 = KK + 1                                                 
            KJ = KK + K                                                 
            IF (N .LT. KP1) GO TO 90                                    
            DO 80 J = KP1, N                                            
               T = AP(KJ)                                               
               AP(KJ) = 0.0D0                                           
               CALL DAXPY(K,T,AP(K1),1,AP(J1),1)                        
               J1 = J1 + J                                              
               KJ = KJ + J                                              
   80       CONTINUE                                                    
   90       CONTINUE                                                    
  100    CONTINUE                                                       
         JJ = 0                                                         
         DO 130 J = 1, N                                                
            J1 = JJ + 1                                                 
            JJ = JJ + J                                                 
            JM1 = J - 1                                                 
            K1 = 1                                                      
            KJ = J1                                                     
            IF (JM1 .LT. 1) GO TO 120                                   
            DO 110 K = 1, JM1                                           
               T = AP(KJ)                                               
               CALL DAXPY(K,T,AP(J1),1,AP(K1),1)                        
               K1 = K1 + K                                              
               KJ = KJ + 1                                              
  110       CONTINUE                                                    
  120       CONTINUE                                                    
            T = AP(JJ)                                                  
            CALL DSCAL(J,T,AP(J1),1)                                    
  130    CONTINUE                                                       
  140 CONTINUE                                                          
      RETURN                                                            
      END                                                               


      SUBROUTINE DPPFA(AP,N,INFO)                                       
      INTEGER N,INFO                                                    
      DOUBLE PRECISION, INTENT(INOUT) :: AP(N*(N+1)/2)
      DOUBLE PRECISION DDOT,T                                           
      DOUBLE PRECISION S                                                
      INTEGER J,JJ,JM1,K,KJ,KK                                          
         JJ = 0                                                         
         DO 30 J = 1, N                                                 
            INFO = J                                                    
            S = 0.0D0                                                   
            JM1 = J - 1                                                 
            KJ = JJ                                                     
            KK = 0                                                      
            IF (JM1 .LT. 1) GO TO 20                                    
            DO 10 K = 1, JM1                                            
               KJ = KJ + 1                                              
               T = AP(KJ) - DDOT(K-1,AP(KK+1),1,AP(JJ+1),1)             
               KK = KK + K                                              
               T = T/AP(KK)                                             
               AP(KJ) = T                                               
               S = S + T*T                                              
   10       CONTINUE                                                    
   20       CONTINUE                                                    
            JJ = JJ + J                                                 
            S = AP(JJ) - S                                              
            IF (S .LE. 0.0D0) GO TO 40                                  
            AP(JJ) = DSQRT(S)                                           
   30    CONTINUE                                                       
         INFO = 0                                                       
   40 CONTINUE                                                          
      RETURN                                                            
      END                                                               


      SUBROUTINE DPPSL(AP,N,B)                                          
      INTEGER N                                                         
      DOUBLE PRECISION AP(1),B(1)                                       
      DOUBLE PRECISION DDOT,T                                           
      INTEGER K,KB,KK                                                   
      KK = 0                                                            
      DO 10 K = 1, N                                                    
         T = DDOT(K-1,AP(KK+1),1,B(1),1)                                
         KK = KK + K                                                    
         B(K) = (B(K) - T)/AP(KK)                                       
   10 CONTINUE                                                          
      DO 20 KB = 1, N                                                   
         K = N + 1 - KB                                                 
         B(K) = B(K)/AP(KK)                                             
         KK = KK - K                                                    
         T = -B(K)                                                      
         CALL DAXPY(K-1,T,AP(KK+1),1,B(1),1)                            
   20 CONTINUE                                                          
      RETURN                                                            
      END                                                               
      subroutine daxpy(n,da,dx,incx,dy,incy)                            
c                                                                       
c     constant times a vector plus a vector.                            
c     uses unrolled loops for increments equal to one.                  
c     jack dongarra, linpack, 3/11/78.                                  
c                                                                       
      double precision dx(1),dy(1),da                                   
      integer i,incx,incy,ix,iy,m,mp1,n                                 
c                                                                       
      if(n.le.0)return                                                  
      if (da .eq. 0.0d0) return                                         
      if(incx.eq.1.and.incy.eq.1)go to 20                               
c                                                                       
c        code for unequal increments or equal increments                
c          not equal to 1                                               
c                                                                       
      ix = 1                                                            
      iy = 1                                                            
      if(incx.lt.0)ix = (-n+1)*incx + 1                                 
      if(incy.lt.0)iy = (-n+1)*incy + 1                                 
      do 10 i = 1,n                                                     
        dy(iy) = dy(iy) + da*dx(ix)                                     
        ix = ix + incx                                                  
        iy = iy + incy                                                  
   10 continue                                                          
      return                                                            
c                                                                       
c        code for both increments equal to 1                            
c                                                                       
c                                                                       
c        clean-up loop                                                  
c                                                                       
   20 m = mod(n,4)                                                      
      if( m .eq. 0 ) go to 40                                           
      do 30 i = 1,m                                                     
        dy(i) = dy(i) + da*dx(i)                                        
   30 continue                                                          
      if( n .lt. 4 ) return                                             
   40 mp1 = m + 1                                                       
      do 50 i = mp1,n,4                                                 
        dy(i) = dy(i) + da*dx(i)                                        
        dy(i + 1) = dy(i + 1) + da*dx(i + 1)                            
        dy(i + 2) = dy(i + 2) + da*dx(i + 2)                            
        dy(i + 3) = dy(i + 3) + da*dx(i + 3)                            
   50 continue                                                          
      return                                                            
      end                                                               
      double precision function ddot(n,dx,incx,dy,incy)                 
c                                                                       
c     forms the dot product of two vectors.                             
c     uses unrolled loops for increments equal to one.                  
c     jack dongarra, linpack, 3/11/78.                                  
c                                                                       
      double precision dx(1),dy(1),dtemp                                
      integer i,incx,incy,ix,iy,m,mp1,n                                 
c                                                                       
      ddot = 0.0d0                                                      
      dtemp = 0.0d0                                                     
      if(n.le.0)return                                                  
      if(incx.eq.1.and.incy.eq.1)go to 20                               
c                                                                       
c        code for unequal increments or equal increments                
c          not equal to 1                                               
c                                                                       
      ix = 1                                                            
      iy = 1                                                            
      if(incx.lt.0)ix = (-n+1)*incx + 1                                 
      if(incy.lt.0)iy = (-n+1)*incy + 1                                 
      do 10 i = 1,n                                                     
        dtemp = dtemp + dx(ix)*dy(iy)                                   
        ix = ix + incx                                                  
        iy = iy + incy                                                  
   10 continue                                                          
      ddot = dtemp                                                      
      return                                                            
c                                                                       
c        code for both increments equal to 1                            
c                                                                       
c                                                                       
c        clean-up loop                                                  
c                                                                       
   20 m = mod(n,5)                                                      
      if( m .eq. 0 ) go to 40                                           
      do 30 i = 1,m                                                     
        dtemp = dtemp + dx(i)*dy(i)                                     
   30 continue                                                          
      if( n .lt. 5 ) go to 60                                           
   40 mp1 = m + 1                                                       
      do 50 i = mp1,n,5                                                 
        dtemp = dtemp + dx(i)*dy(i) + dx(i + 1)*dy(i + 1) +             
     *   dx(i + 2)*dy(i + 2) + dx(i + 3)*dy(i + 3) + dx(i + 4)*dy(i + 4)
   50 continue                                                          
   60 ddot = dtemp                                                      
      return                                                            
      end                                                               


      subroutine  dscal(n,da,dx,incx)                                   
c                                                                       
c     scales a vector by a constant.                                    
c     uses unrolled loops for increment equal to one.                   
c     jack dongarra, linpack, 3/11/78.                                  
c     modified to correct problem with negative increment, 8/21/90.     
c                                                                       
      double precision da,dx(1)                                         
      integer i,incx,ix,m,mp1,n                                         
c                                                                       
      if(n.le.0)return                                                  
      if(incx.eq.1)go to 20                                             
c                                                                       
c        code for increment not equal to 1                              
c                                                                       
      ix = 1                                                            
      if(incx.lt.0)ix = (-n+1)*incx + 1                                 
      do 10 i = 1,n                                                     
        dx(ix) = da*dx(ix)                                              
        ix = ix + incx                                                  
   10 continue                                                          
      return                                                            
c                                                                       
c        code for increment equal to 1                                  
c                                                                       
c                                                                       
c        clean-up loop                                                  
c                                                                       
   20 m = mod(n,5)                                                      
      if( m .eq. 0 ) go to 40                                           
      do 30 i = 1,m                                                     
        dx(i) = da*dx(i)                                                
   30 continue                                                          
      if( n .lt. 5 ) return                                             
   40 mp1 = m + 1                                                       
      do 50 i = mp1,n,5                                                 
        dx(i) = da*dx(i)                                                
        dx(i + 1) = da*dx(i + 1)                                        
        dx(i + 2) = da*dx(i + 2)                                        
        dx(i + 3) = da*dx(i + 3)                                        
        dx(i + 4) = da*dx(i + 4)                                        
   50 continue                                                          
      return                                                            
      end                                                               
                                                                        
