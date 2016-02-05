function xsel_or_grad=bb_select(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    x=args.x;
    idx=args.idx;
    
    xsz=size(x);
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            xsel=x(idx);
            
            xsel_or_grad=xsel;
            
        case 2
            %%% backward pass
            
            D_wrt_xsel=args.D_wrt_thisNode;

            %unique indices in idx
            idxu=unique(idx);
            
            grad=fxns.gpuArray(zeros(xsz));
            for iu=1:length(idxu)
                
                % for each unique index, accumulate incoming gradients to
                % their corresponding element in the gradient
                grad(idxu(iu))=sum(D_wrt_xsel(idx==idxu(iu)));
                
            end
            
            xsel_or_grad=grad;
            
    end
    