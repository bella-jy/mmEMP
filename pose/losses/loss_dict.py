## dict for loss items 
## map the method to its loss components


loss_dict = {
            'cmflow':
                {
                'selfLoss': [],
                'chamferLoss': [],
                'veloLoss':[],
                'smoothnessLoss':[],
                'opticalLoss': [],
                'maskLoss': [],
                'egoLoss': [],
                'motionLoss': [],
                'totalLoss': [],
                },
            'cmflow_t':
                {
                'Loss': [],
                'chamferLoss': [],
                'veloLoss':[],
                'smoothnessLoss':[],
                'egoLoss':[],
                'maskLoss': [],
                'superviseLoss': [],
                'opticalLoss': [],
                },
            'raflow':
                {
                'selfLoss': [],
                'chamferLoss': [],
                'veloLoss':[],
                'smoothnessLoss':[],
                },
            }
