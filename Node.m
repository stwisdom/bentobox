classdef Node < handle
    % A class that implements a node in a computational graph.
    
    properties
        Inputs          % Map of inputs to the node (Labels of parent Nodes)
        Args            % Map of args names, indexed by Inputs
        Output          % stores output of node, updated by FxnForward
        FxnForward      % forward pass function
        Gradient        % stores accumulated gradient w.r.t. to this node
        FxnGradient     % returns gradient of inputs w.r.t. output
        Label           % label for the node
    end
    
    % Node methods
    methods
        % constructor
        function n=Node(opts)
            
            if nargin==0
                % empty constructor case
                opts=struct();
            end
            
            if isfield(opts,'Inputs')
                n.Inputs=opts.Inputs;
            else
                n.Inputs=containers.Map;
            end

            if isfield(opts,'Args')
                n.Args=opts.Args;
            else
                n.Args=containers.Map;
            end
            
            if isfield(opts,'Output')
                n.Output=opts.Output;
            else
                n.Output=[];
            end
            
            if isfield(opts,'FxnForward')
                n.FxnForward=opts.FxnForward;
            else
                n.FxnForward=[];
            end
            
            if isfield(opts,'FxnGradient')
                n.FxnGradient=opts.FxnForward;
            else
                n.FxnGradient=[];
            end
            
            if isfield(opts,'Label')
                n.Label=opts.Label;
            else
                n.Label='';
            end
            
            % initialize Gradient
            
%             n.Gradient=containers.Map;
%             for ii=1:length(n.Inputs)
%                 n.Gradient(n.Inputs{ii})=zeros(size(n.Output));
%             end
            
        end

        
        % runs the node forward
        function runForward(n,args)
            
            n.Output=n.FxnForward(args);

        end
        
        
        % returns gradient of D w.r.t. specified inputLabel
        function grad=runBackward(n,args,inputLabel)
            
            global fxns;
            
            if isempty(n.Gradient)
                % since n.Gradient is empty, this is a cost node, so 
                % initialize gradient to 1.
                n.Gradient=fxns.gpuArray(1);
            end
            
            % pass in this node's gradient of D w.r.t. this node
            args.('D_wrt_thisNode')=n.Gradient;
            % compute gradient of D w.r.t. specified inputLabel
            if isempty(n.FxnGradient)
                error('FxnGradient is not defined for this Node!');
            end
            grad=n.FxnGradient(args,inputLabel);
            
        end
        
        
        % accumulate gradient of D w.r.t. this node into current gradient
        % (grad_wrt_thisNode should be same size as n.Output)
        function accumGradient(n,grad_wrt_thisNode)
            
            if ~isequal(size(grad_wrt_thisNode),size(n.Output))
                error('grad_wrt_thisNode is not the same size as n.Output!');
            end

            if isempty(n.Gradient)
                n.Gradient=grad_wrt_thisNode;
            else
                n.Gradient=n.Gradient+grad_wrt_thisNode;
            end
            
        end
        
%         function runBackward(n,parentLabel_or_childNode,args)
%             
%             global fxns;
%             
%             if isempty(n.Gradient)
%                 % initialize gradient
%                 n.Gradient=fxns.gpuArray(zeros(size(n.Output)));
%             end
%             
%             if ischar(parentLabel_or_childNode)
%                 % parentLabel_or_childNode is a parentLabel
%                 %
%                 % this is a terminal cost node; just compute D w.r.t. the
%                 % specified input given by the label in childNode
%                 n.Gradient=n.FxnGradient(parentLabel_or_childNode,args);
%             else
%                 % parentLabel_or_childNode is a childNode
%                 %
%                 % this is not a terminal node. Job of n.FxnGradient is to 
%                 % combine D w.r.t. childNode with childNode w.r.t. thisNode
%                 % to form D w.r.t. thisNode. The result, D w.r.t.
%                 % thisNode, is accumulated into the current gradient D
%                 % w.r.t. thisNode.
%                 n.Gradient=n.Gradient+n.FxnGradient(parentLabel_or_childNode,args);
%             end
%             
%         end

    end
    
end
