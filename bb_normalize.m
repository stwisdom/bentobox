function xn_or_grad=bb_normalize(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    x=args.x;
    if isfield(args,'dim')
        dim=args.dim;
    else
        dim=2;  %default to normalize across second dimension
    end
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            xn=bsxfun(@rdivide,x,sum(x,dim));
            
            xn_or_grad=xn;
            
        case 2
            %%% backward pass
            
            D_wrt_xn=args.D_wrt_thisNode;
            xn=args.output;
            
            switch(parentLabelGrad)
  
                case 'x'

                    xsum=sum(x,dim);
                    grad=D_wrt_xn.*bsxfun(@minus,1./xsum,bsxfun(@rdivide,xn,xsum));

            end
            
            xn_or_grad=grad;
            
    end
    