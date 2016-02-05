function x_or_grad=bb_whiten(args,parentLabelGrad)
    
    % extract input arguments
    x=args.x;
    mu=args.mu;
    if isfield(args,'std')
        invstd=1./args.std;
    elseif isfield(args,'invstd')
        invstd=args.invstd;
    end
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            x=bsxfun(@times,bsxfun(@minus,x,mu),invstd);
            
            if sum(isinf(x(:)))
                error('Infs in x detected!');
            end
            
            x_or_grad=x;
            
        case 2
            %%% backward pass
            
            D_wrt_x=args.D_wrt_thisNode;
            
            switch(parentLabelGrad)
  
                case 'x'
                    %%%TODO
                    
                case 'mu'
                    %%%TODO
                    
                case 'std'
                    %%%TODO    
                    
                case 'invstd'
                    %%%TODO

            end
            
            x_or_grad=grad;
            
    end
    