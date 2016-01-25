function sm_or_grad=bb_softmax(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    input=args.input;
    if isfield(args,'dim')
        dim=args.dim;
    else
        dim=2;  %default to softmax across second dimension
    end
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            sm=softmax(input,dim);
            
            sm_or_grad=sm;
            
        case 2
            %%% backward pass
            
            D_wrt_sm=args.D_wrt_thisNode;
            sm=args.output;
            
            switch(parentLabelGrad)
  
                case 'input'

                    % for q=sm, derivative of softmax is q(z1)[1-q(z2)] for z1=z2,
                    % q(z1)[-q(z2)] for z1~=z2. This can be written as
                    % q_wrt_logq = diag(q) - qq^T.
                    % Chain rule can be written as
                    % D_wrt_logq = D_wrt_q^T (q_wrt_logq)
                    %            = D_wrt_q^T (diag(q) - qq^T)
                    %            = (D_wrt_q.*q)^T - (D_wrt_q^T q)q^T
                    % compute the element-wise product
                    elem_prod = bsxfun(@times,D_wrt_sm,sm);
                    % compute the chain rule:
                    grad = elem_prod - bsxfun(@times,sum(elem_prod,dim),sm);

            end
            
            sm_or_grad=grad;
            
    end
    