function stdDiag_or_grad=bb_stdDiag(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    x=args.x;
    mu=args.mu;
    dim=args.dim;
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            xsz=size(x);
            
            vDiag=fxns.pagefun_marg(x,x,dim);
            vDiag=vDiag+xsz(dim).*mu.^2;
            vDiag=vDiag-2.*mu.*sum(x,dim);
            vDiag=vDiag./xsz(dim);
            stdDiag=sqrt(vDiag);
            
            stdDiag_or_grad=stdDiag;
            
        case 2
            %%% backward pass
            
            D_wrt_thisNode=args.D_wrt_thisNode;
            stdDiag=args.output;
            
            xsz=size(x); xszlen=length(xsz);
            musz=size(mu); muszlen=length(musz);
            %1-pad dimension vectors:
            if xszlen<muszlen
                xsz=[xsz,ones(1,muszlen-xszlen)];
            else
                musz=[musz,ones(1,xszlen-muszlen)];
            end
            
            switch(parentLabelGrad)
                
                case 'x'
%                     % find dimensions except for dim where x has dim 1 and
%                     % mu has dim greater than 1
%                     xsz_ne_musz=setdiff(find( (xsz==1) && (musz>1) ),dim);
%                     D=D_wrt_thisNode./stdDiag;
%                     if isempty(xsz_ne_musz)
%                         %x and mu are the same size, except for dim
%                         grad=D.*(-mu);
%                     else
%                         Dsum=sum(D,xsz_ne_musz(1));
%                         grad=fxns.pagefun_marg(D,-mu,xsz_ne_musz(1));
%                         if length(xsz_ne_musz)>1
%                             for dd=1:length(xsz_ne_musz)
%                                 Dsum=sum(D,xsz_ne_musz(dd));
%                                 grad=sum(grad,xsz_ne_musz(dd));
%                             end
%                         end
%                     end
%                     
%                     grad=grad+Dsum.*x;

                    grad=bsxfun(@times,D_wrt_thisNode./stdDiag,bsxfun(@minus,x,mu));
                    % find dimensions except for dim where x has dim 1 and
                    % mu has dim greater than 1
                    xsz_ne_musz=setdiff(find( (xsz==1) & (musz>1) ),dim);
                    for dd=xsz_ne_musz
                        grad=sum(grad,dd);
                    end
                    
                case 'mu'
                    grad=bsxfun(@times,D_wrt_thisNode./stdDiag,bsxfun(@minus,mu,x));
                    % find dimensions except for dim where mu has dim 1 and
                    % x has dim greater than 1
                    xsz_ne_musz=find( (musz==1) & (xsz>1) );
                    for dd=xsz_ne_musz
                        grad=sum(grad,dd);
                    end
                    
            end
            
            grad=grad./xsz(dim);
            
            stdDiag_or_grad=grad;
            
    end
    