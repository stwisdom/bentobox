function out=bb_checkGradient(args)

    %required arguments
    bb=args.bb;
    leafNodes=args.leafNodes;
    varCheck =args.varCheck;
    
    %optional arguments
    if isfield(args,'checkDel')
        checkDel=args.checkDel;
    else
        checkDel=1e-6;
    end
    if isfield(args,'checkEps')
        checkEps=args.checkEps;
    else
        checkEps=1e-6;
    end
    if isfield(args,'checkTestProp')
        checkTestProp=args.checkTestProp;
    else
        checkTestProp=0.01;
    end
    if isfield(args,'checkMaxChecks')
        checkMaxChecks=args.checkMaxChecks;
    else
        checkMaxChecks=50;
    end

    
    % perform backwards pass
    fprintf('Performing backwards pass...\n');
    tic;
    bb.clearGradients();
    bb.backwardPass(leafNodes,[],0);
    toc;

    idx=1;
    for ivarCheck=1:length(varCheck)
        
        bb_var=varCheck{ivarCheck};
        if bb.Nodes.isKey(bb_var)
%             % perform backwards pass
%             tic;
%             cn.clearGradients();
%             cn.backwardPass({cn_var});
%             toc;

            % numerical gradient check
            flag_verbose=bb.Flags.verbose;
            bb.Flags.verbose=0;
            outCheck=bb.gradientCheck(struct(...
                                   'startNode',bb_var,...
                                   'del',checkDel,...
                                   'eps',checkEps,...
                                   'testProportion',checkTestProp,...
                                   'maxChecks',checkMaxChecks)...
                                );
            bb.Flags.verbose=flag_verbose;
            if isfield(outCheck,'gradApproxIm')
                %Wirtinger gradient
                gradApprox=outCheck.gradApprox;
            else
                %normal gradient
                gradApprox=outCheck.gradApproxRe;
            end

            mnse=nmse(outCheck.grad,gradApprox);
            if isinf(mnse)
                %sum(target.^2)=0, so just use sum(est.^2):
                mnse=sum(outCheck.grad(:).^2);
            end
            fprintf('NMSE=%e for grad D w.r.t. %s with numerical check.\n\n',mnse,bb_var);
            out{idx}=outCheck;
            out{idx}.gradApprox=gradApprox;
            out{idx}.bb_var=bb_var;
            out{idx}.mnse=mnse;
            idx=idx+1;
        end
        
    end
    