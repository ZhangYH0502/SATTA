import os


def get_args(parser):
    parser.add_argument('--data_root', type=str, default='/research/data/')
    parser.add_argument('--dataset', type=str, choices=['FluidSegDataset'], default='FluidSegDataset')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='results-5/')

    parser.add_argument('--lr', type=float, default=0.001) # 0.00001
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--choose_model', type=str, choices=['UNet', 'UNetModified'], default='UNet')
    parser.add_argument('--num_labels', type=int, default=4)
    parser.add_argument('--inference', action='store_true', default=True)

    parser.add_argument('--name', type=str, default='CTforTrain_SforTest')

    args = parser.parse_args()

    model_name = args.dataset

    model_name += '.'+args.choose_model

    if args.name != '':
        model_name += '.'+args.name
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    model_name = os.path.join(args.results_dir, model_name)
    
    args.model_name = model_name

    if args.inference:
        args.epochs = 1

    if not os.path.exists(args.model_name):
        os.makedirs(args.model_name)

    return args
