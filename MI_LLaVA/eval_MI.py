from evaluation.models.classifier import Classifier
import torch 
import pickle
import os
from evaluation.metrics.classification_acc import ClassificationAccuracy
from final_selection import perform_final_selection
from utils_attack import *

from setproctitle import setproctitle
#################################################### Hyperparam

args = create_parser()
model_name =  args.model_name
isTrain= args.isTrain
skip = args.skip
iterations = args.iterations
samples_per_target = args.samples_per_target
n_imgs = args.n_imgs

config = Config()
config.attack_resize = 336
config.seed = 42
batch_size = 16
approach= 'transforms'
conv_mode = "llava_v1"
prompt_int = "<image>\nWho is the person in the image?"

debug = args.debug
# debug = False
##################################### Load model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_devices = [i for i in range(torch.cuda.device_count())]
stylegan_model = args.stylegan_model
G = load_generator(stylegan_model)
G.to(device)


if stylegan_model =="stylegan2-ada-pytorch/ffhq.pkl":
    config.attack_center_crop = 800
else:
    config.attack_center_crop = None
synthesis = G.synthesis

if debug==True:
    model = None
    tokenizer = None
else:
    model, tokenizer, _ = load_llava_model(model_name)
    model.eval()
    model.to(device)

###########################


data_path = f"inverted/{model_name}_{args.initW_path}_{args.savepath}/"
if args.w =='w':
    save_fol = f"./inverted/{model_name}_{args.initW_path}_{args.savepath}_scores_N{args.iterations}/"
else:        
    save_fol = f"./inverted/{model_name}_{args.initW_path}_{args.savepath}_{args.w}_scores_N{args.iterations}/"

print('data_path',data_path)
if args.dataset =="facescrub":
    with open('metadata/facescrub_idx_to_class.pkl', 'rb') as f:
        idx_to_class = pickle.load(f)
    idx_to_class[93] = 'Freddy_Prinze_Jr_'
    idx_to_class[107] = 'Harry_Connick_Jr_'
    idx_to_class[238] = 'Robert_Downey_Jr_'
elif args.dataset == "celeba":
    with open('metadata/celeba_idx_to_class.pkl', 'rb') as f:
        idx_to_class = pickle.load(f)
elif args.dataset =="StanfordDogs":  
    with open('metadata/stanford_dogs_idx_to_class.pkl', 'rb') as f:
        idx_to_class = pickle.load(f)
    
    prompt_int = "<image>\nWhat breed is this dog?"
w_optimized_unselected = []
targets = []

for gt in idx_to_class.keys():
    if gt<args.evalID:        
        print(gt,idx_to_class[gt])
        folder = idx_to_class[gt]
        folder = os.path.join(data_path,folder)
        # print('folder',folder)
        w = []
        try:
            for i in range(n_imgs):
                data_ = torch.load(f'{folder}/{i}.pt')
                w_ = data_[args.w]
                w.append(w_)
            w = torch.cat((w),dim=0)
            w_optimized_unselected.append(w)
            targets.append(torch.zeros(w.shape[0],dtype=torch.int64)+gt)
        except BaseException:
            # Code that runs if an exception occurs
            print("-------",folder)
w_optimized_unselected = torch.cat(w_optimized_unselected, dim=0)
targets = torch.cat(targets, dim=0)
print('w_optimized_unselected',w_optimized_unselected.shape)
print('targets',targets.shape)

os.makedirs(save_fol, exist_ok=True)
final_w, final_targets = perform_final_selection(
            w_optimized_unselected,
            synthesis,
            config,
            targets,
            model,
            samples_per_target,
            approach,
            iterations,
            idx_to_class,
            conv_mode,
            prompt_int,
            tokenizer,
            device,
            skip,          
            save_fol 
            )

optimized_w_path_selected = f"{data_path}/optimized_w_selected_{args.w}.pt"
torch.save({'w': final_w.detach(), 'targets': final_targets.detach()}, optimized_w_path_selected)

del model
torch.cuda.empty_cache()
# load evaluation model
if args.dataset =="facescrub":
    evaluation_model = Classifier(num_classes=530,  architecture=' inception-v3')
    evaluation_model.load_state_dict(torch.load('evaluation_models/facescrub/inception_v3_facescrub.pt'))
    
elif args.dataset =="celeba":    
    evaluation_model = Classifier(num_classes=1000,  architecture=' inception-v3')
    evaluation_model.load_state_dict(torch.load('evaluation_models/celeba/inception_v3_celeba.pt'))

elif args.dataset =="StanfordDogs":    
    evaluation_model = Classifier(num_classes=120,  architecture=' inception-v3')
    evaluation_model.load_state_dict(torch.load('evaluation_models/stanford_dogs/inception_v3_stanford_dogs.pt'))

evaluation_model = torch.nn.DataParallel(evaluation_model)
evaluation_model.to(device)
evaluation_model.eval()
class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                     device=device)

if args.unfiltered:
    
    setproctitle(f'Unfiltered Evaluation of {final_w.shape[0]*2} images on Inception-v3')
    
    acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
                w_optimized_unselected,
                targets,
                synthesis,
                config,
                batch_size=batch_size * 2,
                resize=299,
                rtpt=None)        
    print(
        f'\nUnfiltered Evaluation of {w_optimized_unselected.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}',
        f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
    )
    write_precision_list(f"{data_path}/precision_list_unfiltered_{args.w}", precision_list)

    # Save unfiltered results to text file
    with open(f"{data_path}/evaluation_results_{args.w}.txt", "a") as f:
        f.write(f'\nUnfiltered Evaluation of {w_optimized_unselected.shape[0]} images on Inception-v3:\n')
        f.write(f'accuracy@1={acc_top1:4f}, accuracy@5={acc_top5:4f}\n')
        f.write(f'correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}\n\n')
    # exit()

setproctitle(f'Filtered Evaluation of {final_w.shape[0]} images on Inception-v3')
acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
            final_w,
            final_targets,
            synthesis,
            config,
            batch_size=batch_size * 2,
            resize=299,
            rtpt=None)

print(
    f'Filtered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}, ',
    f'accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
)

# Append filtered results to text file
with open(f"{data_path}/evaluation_results_{args.w}.txt", "a") as f:
    f.write(f'Filtered Evaluation of {final_w.shape[0]} images on Inception-v3, IDs from {skip[0]} to {skip[1]}:\n')
    f.write(f'accuracy@1={acc_top1:4f}, accuracy@5={acc_top5:4f}\n')
    f.write(f'correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}\n')

write_precision_list(f"{data_path}/precision_list_filtered_{args.w}", precision_list)

del class_acc_evaluator, evaluation_model

torch.cuda.empty_cache()
####################################
#         Feature Distance         #
####################################

setproctitle(f'Feature Distance on Inception-v3')
avg_dist_inception = None
avg_dist_facenet = None
# Load Inception-v3 evaluation model and remove final layer
# evaluation_model_dist = config.create_evaluation_model()
if args.dataset =="facescrub":
    evaluation_model_dist = Classifier(num_classes=530,  architecture=' inception-v3')
    evaluation_model_dist.load_state_dict(torch.load('evaluation_models/facescrub/inception_v3_facescrub.pt'))
    
elif args.dataset =="celeba":    
    evaluation_model_dist = Classifier(num_classes=1000,  architecture=' inception-v3')
    evaluation_model_dist.load_state_dict(torch.load('evaluation_models/celeba/inception_v3_celeba.pt'))

elif args.dataset =="StanfordDogs":    
    evaluation_model_dist = Classifier(num_classes=120,  architecture=' inception-v3')
    evaluation_model_dist.load_state_dict(torch.load('evaluation_models/stanford_dogs/inception_v3_stanford_dogs.pt'))


evaluation_model_dist.model.fc = torch.nn.Sequential()
evaluation_model_dist = torch.nn.DataParallel(evaluation_model_dist,
                                                device_ids=gpu_devices)
evaluation_model_dist.to(device)
evaluation_model_dist.eval()

target_dataset = args.dataset
batch_size_single = 16
from metrics.distance_metrics import DistanceEvaluation
# Compute average feature distance on Inception-v3
evaluate_inception = DistanceEvaluation(evaluation_model_dist,
                                        synthesis, 299,
                                        config.attack_center_crop,
                                        target_dataset, config.seed)
avg_dist_inception, mean_distances_list = evaluate_inception.compute_dist(
    final_w,
    final_targets,
    batch_size=batch_size_single * 5)


print('Mean Distance on Inception-v3: ',
        avg_dist_inception.cpu().item())


# Append Mean Distance on Inception-v3 to text file
with open(f"{data_path}/evaluation_results_{args.w}.txt", "a") as f:
    f.write(f'\nMean Distance on Inception-v3: {avg_dist_inception.cpu().item()}\n')

from facenet_pytorch import InceptionResnetV1
# Compute feature distance only for facial images
if target_dataset in [
        'facescrub', 'celeba'
]:
    
    setproctitle(f'Feature Distance on FaceNet')
    # Load FaceNet model for face recognition
    facenet = InceptionResnetV1(pretrained='vggface2')
    facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
    facenet.to(device)
    facenet.eval()

    # Compute average feature distance on facenet
    evaluater_facenet = DistanceEvaluation(facenet, synthesis, 160,
                                            config.attack_center_crop,
                                            target_dataset, config.seed)
    avg_dist_facenet, mean_distances_list = evaluater_facenet.compute_dist(
        final_w,
        final_targets,
        batch_size=batch_size_single * 8)
    
    print('Mean Distance on FaceNet: ', avg_dist_facenet.cpu().item())
    # Append Mean Distance on Inception-v3 to text file
    with open(f"{data_path}/evaluation_results_{args.w}.txt", "a") as f:
        f.write(f'\nMean Distance on FaceNet: {avg_dist_facenet.cpu().item()}\n')


