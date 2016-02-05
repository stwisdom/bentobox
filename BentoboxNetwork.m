classdef BentoboxNetwork < handle %
% A bespoke network that can be discriminatively trained.
    
    % BentoboxNetwork properties
    properties  
        Flags           %flags, struct of Booleans
        Nodes           %Map of Node objects, Map(string,Node)
        NodeOrder       %cell array of Node keys defining forward computation order
        NodesChildren   %Map of each Node's children, Map(string,cell of strings)
        AdjMtx          %adjacency matrix of Nodes, sparse numNodes x numNodes Boolean matrix
        AdjLabels       %ordered list of Node labels corresponding to indices in AdjMtx (cell array)
    end
       
    
    % BentoboxNetwork methods
    methods
        % constructor
        function cn = BentoboxNetwork(opts)
            
            if nargin==0
                % empty constructor
                opts=struct();
            end
            
            
            %%% set Flags
            if isfield(opts,'flag_verbose')
                cn.Flags.verbose=opts.flag_verbose;
            else
                cn.Flags.verbose=1;
            end
            
            
            %%% set Nodes
            if isfield(opts,'Nodes')
                cn.print('Initializing computational graph with supplied Nodes\n');
                cn.Nodes=opts.Nodes;
            else
                cn.print('Initializing computational graph with no Nodes.\n');
                cn.Nodes=containers.Map;
            end
            
            
            %%% set NodesChildren, AdjMtx, and AdjLabels
            keys=cn.Nodes.keys;
            cn.NodesChildren=containers.Map;
            cn.AdjMtx=sparse(length(keys),length(keys));
            cn.AdjLabels=keys;
            % inital iteration through all Nodes:
            for ii=1:length(keys)
                cn.NodesChildren(keys{ii})={};
            end
            % iterate through Nodes, finding children
            for ii=1:length(keys)
                inputs=cn.Nodes(keys{ii});
                for jj=1:length(inputs)
                    cn.NodesChildren(keys{ii})=cat(1,cn.NodesChildren(keys{ii}),inputs{jj});
                    idxSink=find(ismember(keys, inputs{jj}));
                    cn.AdjMtx(idxSink,ii)=true;
                end
            end
            
            
            %%% set NodeOrder
            if isfield(opts,'NodeOrder')
                cn.NodeOrder=opts.NodeOrder;
            else
                if isempty(cn.Nodes)
                    % no Nodes added yet
                    cn.NodeOrder={};
                else
                    % just set NodeStart to first key
                    keys=cn.Nodes.keys;
                    cn.NodeOrder=keys;
                end
            end
            
        end
        
        
        % add a node
        % this added node will become the last node
        function addNode(cn,Node)
            cn.print(sprintf('Adding %s to network.\n',Node.Label));
            cn.Nodes(Node.Label)=Node;
            
            % set relevant entry in NodesChildren
            cn.NodesChildren(Node.Label)={};
            % add dimension to AdjMtx:
            [i,j,s] = find(cn.AdjMtx);
            n = size(cn.AdjMtx,1);
            cn.AdjMtx=sparse(i,j,s,n+1,n+1);
            % add label to AdjLabels:
            cn.AdjLabels{end+1}=Node.Label;
            
            % get inputs of this new node
            inputs=Node.Inputs;
            % get the keys corresponding to entries in the AdjMtx
            keys=cn.AdjLabels;
            % for each input of this new node,
            for jj=1:length(inputs)
                % add this new node to the list of children for its parent node
                cn.NodesChildren(inputs{jj})=cat(1,cn.NodesChildren(inputs{jj}),Node.Label);
                % update the AdjMtx
                idxSink=find(ismember(keys, inputs{jj}));
                cn.AdjMtx(idxSink,n+1)=true;
            end
            
            if isempty(cn.NodeOrder)
                % no NodeOrder list exists; create it, setting the current
                % node as the first node
                cn.NodeOrder{1}=Node.Label;
            else
                % append the current node to the NodeOrder list
                cn.NodeOrder{end+1}=Node.Label;
            end
        end
        
        
        % perform forward pass of network
        function forwardPass(cn,startNode,flag_recomputeInputs,nodeOrder)
            
            if ~exist('nodeOrder','var') || isempty(nodeOrder)
                % default nodeOrder to be the cn's NodeOrder
                nodeOrder=cn.NodeOrder;
            end
            
            if ~exist('startNode','var')
                % by default, start from first node in NodeOrder
                startNode=cn.NodeOrder{1};
                startNodeIdx=1;
            else
                startNodeIdx=stridx(startNode,nodeOrder);
            end
            
            if ~exist('flag_recomputeInputs','var')
                % default to always recomputing inputs
                flag_recomputeInputs=1;
            end

            cn.print(sprintf('Running forward pass from %s, %d of %d...\n',startNode,startNodeIdx,length(nodeOrder)));

            if flag_recomputeInputs
                % create Map of flags indicating if nodes have been
                % computed, used to avoid computing variables multiple
                % times.
                flagsComputed=containers.Map(cn.NodeOrder,zeros(1,length(cn.NodeOrder)));
            end
            
            % iterate through the nodes
            for inode=startNodeIdx:length(nodeOrder)
                keyCur=nodeOrder{inode}; %current Node Label
                if inode>1
                    % check that all input nodes for this node have been
                    % computed
                    keysInput=cn.Nodes(keyCur).Inputs;

                    if cn.Flags.verbose
                        % if we're being verbose, report names of parent
                        % nodes
                        if ~isempty(keysInput)
                            nodesDep='';
                            for ii=1:length(keysInput)
                                nodesDep=[nodesDep,keysInput{ii},', '];
                            end
                            nodesDep(end-1:end)='. ';
                        else
                            nodesDep='nothing.';
                        end
                        cn.print(sprintf('  Computing %s, depends on %s\n',keyCur,nodesDep));
                    end
                    
                    if flag_recomputeInputs
                        for ii=1:length(keysInput)
                            keyInput=keysInput{ii};
                            if ~flagsComputed(keysInput{ii})
                                cn.Nodes(keyInput).runForward(cn.collectInputs(keyInput));
                                flagsComputed(keyInput)=1;
                            end
                        end
                    end
                end
                
                NodeCur=cn.Nodes(keyCur);   %get the current Node
                NodeCur.runForward(cn.collectInputs(keyCur));   %pass Inputs to Node's forward function
                cn.Nodes(keyCur)=NodeCur;   %assign updated Node
                flagsComputed(keyCur)=1;    %this Node has been computed
                cn.print(sprintf('  Computed %s, %d of %d.\n',keyCur,inode,length(nodeOrder)));
            end

        end
        
        
        % compute gradients from cost function (assumed to be last node) to
        % nodes in leafList
        function backwardPass(cn,leafList,nodeOrderList,flag_recompute)
            
            numLeaves=length(leafList);
            
            if ~exist('nodeOrderList','var') || isempty(nodeOrderList)
                % default to no manually-specified paths
                nodeOrderList=cell(1,numLeaves);
            end
            
            if ~exist('flag_recompute','var')
                %flag_recompute=1;
                flag_recompute=0;
            end
            
            % find indices of leaves in cn.AdjLabels, the list of adjacency
            % labels
            leafIdx=zeros(1,numLeaves);
            for nn=1:length(leafList)
                leafIdx(nn)=stridx(leafList{nn},cn.AdjLabels);
            end
            % order the nodes in leafList from deepest (smallest index) to
            % shallowest (largest index)
            leafIdxSorted=sort(leafIdx,'ascend');
            
            % find path from last node (assumed to be cost D) to input
            toD=graphtraverse(cn.AdjMtx',length(cn.AdjLabels));
            for nn=1:numLeaves
                % find path from current leaf to last node (assumed D)
                fromLeaf=graphtraverse(cn.AdjMtx,leafIdxSorted(nn));
                % path from current leaf to D
                leafToD=intersect(fromLeaf,toD);
                
                if ~isempty(nodeOrderList{nn})
                    % if a manual nodeOrder is specified, intersect it with
                    % the current nodeOrder, leafToD
                    nodeOrderListIdx=stridx(nodeOrderList{nn},cn.AdjLabels);
                    leafToD=intersect(leafToD,nodeOrderListIdx);
                end
                
                cn.print(sprintf('  Starting backwards pass to %s...\n',cn.AdjLabels{leafIdxSorted(nn)}));
                
                % iterate downwards, with the first parent being the parent
                % of cost node D (i.e., start at second-to-last node in
                % path leafToD)
                for iparent=(length(leafToD)-1):-1:1
                    
                    cn.print(sprintf('  Backwards pass to %s.\n',cn.AdjLabels{leafToD(iparent)}));
                    
                    % get current parent Node.Label
                    parentCurLabel=cn.AdjLabels{leafToD(iparent)};
                    % get current parent Node
                    parentCur=cn.Nodes(parentCurLabel);
                    
                    % check if we need to compute the gradients
                    if ~(isempty(parentCur.Gradient) || flag_recompute)
                        cn.print(sprintf('  Gradient w.r.t. %s already computing, continuing...\n',cn.AdjLabels{leafToD(iparent)}));
                        continue;
                    end
                    
                    % get list of parent Node's children
                    childrenCurLabels=cn.NodesChildren(parentCurLabel);
                    % find indices of current children
                    childrenCurIdx=stridx(childrenCurLabels,cn.AdjLabels);
                    % intersect path from current leaf to D with current
                    % children, to make sure no extra computation is
                    % performed
                    childrenCurIdx=intersect(childrenCurIdx,leafToD);
                    % convert children indices back to cell array of
                    % strings
                    childrenCurLabels=cn.AdjLabels(childrenCurIdx);
                    % for each child of the current parent, run its
                    % backward pass to the parent:
                    for ichild=1:length(childrenCurLabels)
                        
                        % get current child node
                        childCur=cn.Nodes(childrenCurLabels{ichild});
                        
                        cn.print(sprintf('  Passing gradient backwards from child %s to parent %s.\n',childCur.Label,cn.AdjLabels{leafToD(iparent)}));
                        
                        % get arguments to run child node backwards to
                        % parent:
                        args=cn.collectInputs(childCur.Label);
                        args.('output')=childCur.Output;
                        % convert current parent's label to an arg name
                        parentArgName=childCur.Args(parentCurLabel);
                        % run child node backwards to parent
                        grad_wrt_parent=childCur.runBackward(args,parentArgName);
                        
                        % accumulate gradient into parent
                        if strmatch(parentCur.Label,'logLike','exact')
                            stuff=1;
                        end
                        parentCur.accumGradient(grad_wrt_parent);
                        
                    end
                    
                    % update the parent node in the CN
                    cn.Nodes(parentCurLabel)=parentCur;
                    
                end
            end
            
        end
        
        
        % clear the current gradients (important to do for a backwards pass
        % to a different variable)
        function clearGradients(cn)
            
            keys=cn.Nodes.keys;
            for nn=1:length(keys)
                curNode=cn.Nodes(keys{nn});
                curNode.Gradient=[];
                cn.Nodes(keys{nn})=curNode;
            end
            
        end
        
        
        % check a gradient
        function out=gradientCheck(cn,args)
            
            % load arguments
            startNodeLabel=args.startNode;
            del=args.del;
            if isfield(args,'eps')
                eps=args.eps;
            end
            if isfield(args,'testProportion')
                testProportion=args.testProportion;
            else
                testProportion=1;
            end
            if isfield(args,'nodeOrderList')
                nodeOrder=args.nodeOrderList;
            end
            
            % get startNode
            startNode=cn.Nodes(startNodeLabel);
            % set independent variable to output of startNode
            varIndep=startNode.Output;
            
            if isfield(args,'maxChecks')
                % a maximum number of numerical gradient checks is
                % specified; change testProportion if necessary
                maxChecks=args.maxChecks;
                NtestProportion=floor(testProportion*numel(varIndep));
                if (NtestProportion==0)
                    NtestProportion=numel(varIndep);
                end
                testProportion=min(NtestProportion,maxChecks)/numel(varIndep);
            end
            
            if isfield(args,'flag_nonneg')
                flag_nonneg=args.flag_nonneg;
            else
                flag_nonneg=0;
            end
            
            % arguments to partial forwardPass function
            args=struct('cn',cn,'startNode',startNodeLabel);
            if exist('nodeOrder','var')
                % if a manual node order is specified, pass it to
                % bb_forwardPass_partial
                args.nodeOrder=nodeOrder;
            end
            % define partial forward pass function
            fxn_update_dep=@(varIndep,args)bb_forwardPass_partial(varIndep,args);
            
            % perform numerical gradient checks
            [gradApproxRef,jgradApproxImf,itest]=gradientCheck(varIndep,fxn_update_dep,args,del,eps,testProportion,flag_nonneg);
            
            gradApproxf=gradApproxRef;
            if isempty(jgradApproxImf)
                gradApproxf=0.5.*gradApproxRef;
            else
                % need to subtract for Wirtinger gradients:
                gradApproxf=0.5.*(gradApproxf-jgradApproxImf);
            end

            startNodeGrad=startNode.Gradient;
            
            startNodeGradVec=startNodeGrad(:);
            gradErr=startNodeGradVec(itest(:))-gradApproxf(:);
            fprintf('Errors for D w.r.t. %s\n',startNodeLabel);
            meanAbsGradErr = mean(abs(gradErr(:)))
            totalAbsGradErr = sum(abs(gradErr(:)))
            maxAbsGradErr = max(abs(gradErr(:)))
            
            out.gradApproxRe=gradApproxRef(:).';
            out.jgradApproxIm=jgradApproxImf(:).';
            out.gradApprox=gradApproxf(:).';
            out.grad=startNodeGrad(itest(:)).';
            out.itest=itest(:).';
            
        end
            

        % collect the inputs for a Node and return a struct
        % (used to pass in Node.fxnForward's arguments)
        function args=collectInputs(cn,key)
            keysInput=cn.Nodes(key).Inputs;
            args=struct();
            for ii=1:length(keysInput)
                keyInput=keysInput{ii};
                if ~isempty(cn.Nodes(key).Args)
                    Args=cn.Nodes(key).Args;
                    argName=Args(keyInput);
                else
                    argName=keyInput;
                end
                args.(argName)=cn.Nodes(keyInput).Output;
            end
        end
        
        
        % printing function
        function print(cn,string)
            if cn.Flags.verbose
                fprintf([' BentoboxNetwork: ' string]);
            end
        end
            
    end
        
end
