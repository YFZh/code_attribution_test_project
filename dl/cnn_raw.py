class CNN(nn.Module):
    """CNN."""

    def __init__(self, model_arch="resnet50", n_classes=2, include_top=False, pretrained=False, lower_features=False):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.include_top = include_top
        self.pretrained = pretrained
        self.lower_features = lower_features

        self.gradients = None
        self.classifier = None

        if (model_arch == "resnet50"):
            self.model = models.resnet50(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, n_classes)

            self.features = nn.Sequential(*list(self.model.children())[:-1])
            #print(self.features)
            self.features_dict = OrderedDict()

        elif (model_arch == "resnet101"):
            self.model = models.resnet101(pretrained=True)
            #print(self.model)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, n_classes)

            self.features_dict = OrderedDict()
            if (lower_features == True):
                self.model = nn.Sequential(*list(self.model.children())[:5])
            else:
                self.model = nn.Sequential(*list(self.model.children())[:-2])

        elif (model_arch == "squeezenet"):
            self.model = models.squeezenet1_1(pretrained=True)
            #print(self.model)
            #self.classifier = self.model.classifier

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            #num_ftrs = self.model.fc.in_features
            #self.model.fc = torch.nn.Linear(num_ftrs, n_classes)
            #num_ftrs = 512
            #self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            self.features_dict = OrderedDict()
            if (lower_features == True):
                self.model = nn.Sequential(self.model.features[:6])
            else:
                self.model = nn.Sequential(self.model.features)

            #print(self.model)
            #exit()
        elif (model_arch == "densenet121"):
            self.model = models.densenet121(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(num_ftrs, n_classes)
            self.features = nn.Sequential(*list(self.model.children())[:-1])
            print(self.model)

        elif (model_arch == "vgg19"):
            self.model = models.vgg19(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier[0].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            self.features = nn.Sequential(*list(self.model.children())[:-1])
            #print(self.features)
            print(self.model)

        elif (model_arch == "vgg16"):
            self.model = models.vgg16(pretrained=True);

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier[0].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            if(lower_features == True):
                self.model = nn.Sequential(self.model.features[:5])
            else:
                self.model = nn.Sequential(*list(self.model.children())[:-2])

            #print(self.features)
            #print(self.model)
            #exit()
            print(self.model)
            self.features_dict = OrderedDict()

        elif (model_arch == "mobilenet"):
            self.model = models.mobilenet_v2(pretrained=True);

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            if(lower_features == True):
                #self.model = nn.Sequential(self.model.features[:5])
                self.model = nn.Sequential(*list(self.model.features)[:5])
            else:
                #self.model = nn.Sequential(*list(self.model.children())[:-1])
                self.model = nn.Sequential(*list(self.model.features))

            self.features_dict = OrderedDict()

        elif (model_arch == "alexnet"):
            self.model = models.alexnet(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier[0].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            self.features = nn.Sequential(*list(self.model.children())[:-1])
            #print(self.features)
            print(self.model)

        else:
            self.model_arch = None
            print("No valid backbone cnn network selected!")

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def forward(self, x):
        """Perform forward."""
        if(self.include_top == False):
            # extract features
            x = self.model(x)
            self.features_dict['out'] = x
            self.features_dict['aux'] = x
            return self.features_dict

        elif(self.include_top == True):
            #print(x.size())
            x = self.model(x)

            # flatten
            x = x.view(x.size(0), -1)

            x = self.classifier(x)
            self.features_dict['out'] = x

            return self.features_dict

        return x
    
    def __str__(self):
        return "CNN"