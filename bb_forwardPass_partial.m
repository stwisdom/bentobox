function D=bb_forwardPass_partial(startNodeOutput,args)

    % get arguments
    cn=args.cn;
    startNodeLabel=args.startNode;
    
    if isfield(args,'nodeOrder')
        % cell array of cell arrays
        nodeOrder=args.nodeOrder;
    else
        nodeOrder=cn.NodeOrder;
    end

    % get index of startNode in NodeOrder
    startNodeIdx=stridx(startNodeLabel,nodeOrder);
    % get label of Node after startNode
    afterStartNodeLabel=nodeOrder{1+startNodeIdx};
    NodesOutputSave=cell(1,length(nodeOrder)-startNodeIdx+1);
    
    % save off Outputs of Nodes that will be perturbed
    idx=startNodeIdx;
    for ii=1:length(NodesOutputSave)
        curNode=cn.Nodes(nodeOrder{idx});
        NodesOutputSave{ii}=curNode.Output;
        idx=idx+1;
    end
    
    % set startNode's output to the perturbed value
    startNode=cn.Nodes(startNodeLabel);
    startNode.Output=startNodeOutput;
    cn.Nodes(startNodeLabel)=startNode;

    % run the network forward and get the cost with perturbed output
    Dorig=cn.Nodes(nodeOrder{end}).Output;
    cn.forwardPass(afterStartNodeLabel,0,nodeOrder);
    D=cn.Nodes(nodeOrder{end}).Output;
    
    % restore Outputs of perturbed Nodes to original values
    idx=startNodeIdx;
    for ii=1:length(NodesOutputSave)
        curNode=cn.Nodes(nodeOrder{idx});
        curNode.Output=NodesOutputSave{ii};
        cn.Nodes(nodeOrder{idx})=curNode;
        idx=idx+1;
    end
    