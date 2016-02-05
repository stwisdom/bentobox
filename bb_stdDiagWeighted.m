function stdDiag_or_grad=bb_stdDiagWeighted(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    x=args.x;
    mu=args.mu;
    w=args.w;
    dim=args.dim;
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            xsz=size(x);

            vDiag=fxns.pagefun_marg(w,x.^2,dim);
            vDiag=bsxfun(@plus,vDiag,bsxfun(@times,sum(w,dim),mu.^2));
            vDiag=bsxfun(@minus,vDiag,-2.*bsxfun(@times,fxns.pagefun_marg(w,x,dim),mu));
            stdDiag=sqrt(vDiag);
            
            stdDiag_or_grad=stdDiag;
            
        case 2
            %%% backward pass
            
            D_wrt_thisNode=args.D_wrt_thisNode;
            stdDiag=args.output;
            
            xsz=size(x); xszlen=length(xsz);
            musz=size(mu); muszlen=length(musz);
            wsz=size(w); wszlen=length(wsz);
            %1-pad dimension vectors:
            if xszlen<muszlen
                xsz=[xsz,ones(1,muszlen-xszlen)];
            else
                musz=[musz,ones(1,xszlen-muszlen)];
            end
            if wszlen<muszlen
                wsz=[wsz,ones(1,wszlen-muszlen)];
            else
                musz=[musz,ones(1,wszlen-muszlen)];
                xsz=[xsz,ones(1,wszlen-muszlen)];
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

                    grad=D_wrt_thisNode./stdDiag.*bsxfun(@minus,x,mu);
                    grad=bsxfun(@times,grad,w);
                    % find dimensions except for dim where x has dim 1 and
                    % mu has dim greater than 1
                    xsz_ne_musz=setdiff(find( (xsz==1) && (musz>1) ),dim);
                    for dd=xsz_ne_musz
                        grad=sum(grad,dd);
                    end
                    
                case 'mu'
                    grad=D_wrt_thisNode./stdDiag.*bsxfun(@minus,mu,x);
                    grad=bsxfun(@times,grad,w);
                    % find dimensions except for dim where mu has dim 1 and
                    % x has dim greater than 1
                    xsz_ne_musz=setdiff(find( (musz==1) && (xsz>1) ),dim);
                    for dd=xsz_ne_musz
                        grad=sum(grad,dd);
                    end
                    
                case 'w'
                    grad=bsxfun(@times,0.5.*D_wrt_thisNode./stdDiag,bsxfun(@minus,bsxfun(@plus,x.^2,mu.^2),2.*bsxfun(@times,x,mu)));
                    wsz_ne_musz_xsz=setdiff(find( (wsz==1) && (musz>1 || xsz>1) ),dim);
                    for dd=wsz_ne_musz_xsz
                        grad=sum(grad,dd);
                    end
                    
            end
            
            stdDiag_or_grad=grad;
            
    end
    