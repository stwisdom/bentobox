clear variables;
addpath('../');

%% load data
  [X,Y]=load_data();
  %X is Nxd, where N is number of examples and d is ambient dimension
  [N,d]=size(X);
  %Y is NxC binary, where C is number of classes
  [~,C]=size(Y);

%% network parameters
  Nlayers=3;      %number of layers in the network
  nonlin='ReLU';  %type of nonlinearity
  Nhidden=2048;   %number of hidden nodes

%% initialize the bentobox network:
  opts=struct('flag_verbose',0);
  bb=BentoboxNetwork(opts);

%% add nodes to the bentobox network
  %input data
  n=Node('Label','X');
  n.FxnForward=@(args)X;
  bb.addNode(n);
  
  %input layer
  %  linear part of affine transform
  n=Node('Label','A_0');
  n.FxnForward=@(args)Ainit;
  bb.addNode(n);
  %  bias in affine transform
  n=Node('Label','b_0');
  n.FxnForward=@(args)binit;
  bb.addNode(n);
  
  %  affine transform of data
  n=Node('Label','aff_0');
  n.Inputs={'X',...
            'A_0',...
            'b_0'};
  n.Args(n.Inputs{1})='X';
  n.Args(n.Inputs{2})='A';
  n.Args(n.Inputs{3})='b';
  n.FxnForward=@(args)bb_affine(args);
  n.FxnGradient=@(args,parentLabel)bb_affine(args,parentLabel);
  bb.addNode(n);
  
  %  nonlinearity in input layer
  n=Node('Label','Z_0');
  n.Inputs={'aff_0'};
  n.Args(n.Inputs{1})='input';
  n.FxnForward=@(args)bb_nonlin(setfield(args,'type',nonlin));
  n.FxnGradient=@(args,parentLabel)bb_nonlin(...
                                     setfield(args,'type',nonlin),...
                                     parentLabel...
                                   );
  %build the network layers
  for ilayer=1:Nlayers
      
      % linear part of affine transform
      n=Node('Label',sprintf('A_%d',ilayer));
      n.FxnForward=@(args)Ainit;
      bb.addNode(n);
      % bias in affine transform
      n=Node('Label',sprintf('b_%d',ilayer));
      n.FxnForward=@(args)binit;
      bb.addNode(n);
      
      % affine transformation of data
      n=Node('Label',sprintf('aff_%d',ilayer));
      n.Inputs={sprintf('Z_%d',ilayer-1),...
                sprintf('A_%d',ilayer),...
                sprintf('b_%d',ilayer)};
      n.Args(n.Inputs{1})='X';
      n.Args(n.Inputs{2})='A';
      n.Args(n.Inputs{3})='b';
      n.FxnForward=@(args)bb_affine(args);
      n.FxnGradient=@(args,parentLabel)bb_affine(args,parentLabel);
      bb.addNode(n);
      
      % nonlinearity
      n=Node('Label',sprintf('Z_%d',ilayer));
      n.Inputs={sprintf('aff_%d',ilayer)};
      n.Args(n.Inputs{1})='input';
      n.FxnForward=@(args)bb_nonlin(setfield(args,'type',nonlin));
      n.FxnGradient=@(args,parentLabel)bb_nonlin(...
                                         setfield(args,'type',nonlin),...
                                         parentLabel...
                                       );
      
  end
  
  %build cost function
  %  output layer of network
  n=Node('Label','Yhat');
  n.Inputs={sprintf('Z_%d',Nlayers)};
  n.Args(n.Inputs{1})='input';
  n.FxnForward=@(args)bb_softmax(args);
  n.FxnGradient=@(args,parentLabel)bb_softmax(args,parentLabel);
  bb.addNode(n);
  %  target labels
  n=Node('Label','Y');
  n.FxnForward=Y;
  bb.addNode(n);
  %  cost function
  n=Node('Label','D_CE');
  n.Inputs={'Y','Yest'};
  n.Args(n.Inputs{1})='Y';
  n.Args(n.Inputs{2})='Yhat';
  n.FxnForward=@(args)bb_crossEntropy(args);
  n.FxnGradient=@(args)bb_crossEntropy(args,parentLabel);
  bb.addNode(n);
  
  %% run the network forward
  bb.forwardPass();
  
  %% get gradients w.r.t. parameters
  % build a list of nodes to take gradients to
  leafList={};
  for ilayer=0:Nlayers
      leafList={leafList,sprintf('A_%d',ilayer)};
      leafList={leafList,sprintf('b_%d',ilayer)};
  end
  % compute gradients of D_CE w.r.t. nodes in leafList
  bb.backwardPass(leafList);
  
  %% example calls 
  % extract an intermediate node's value, e.g. Z_<Nlayers>:
  Zlast=bb.Nodes(sprintf('Z_%d',Nlayers)).Output;
  % extract a gradient, e.g. A_2:
  D_wrt_A2=bb.Nodes('A_2').Gradient;
  