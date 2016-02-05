function mu_or_grad=bb_mean(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    x=args.x;
    dim=args.dim;
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            mu=mean(x,dim);
            
            mu_or_grad=mu;
            
        case 2
            %%% backward pass
            
            D_wrt_thisNode=args.D_wrt_thisNode;
            
            xsz=size(x);
            grad=bsxfun(@times,D_wrt_thisNode,fxns.gpuArray(ones(xsz)))./xsz(dim);
            
            mu_or_grad=grad;
            
    end
    