import os
import random
import sys
import json
import codecs as cs
from tqdm import tqdm
from os.path import join as pjoin
sys.argv = ['GPT_eval_multi.py']
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
import os
#os.environ['PYOPENGL_PLATFORM'] = 'pyglet'

args.dataname = 't2m'
args.resume_pth = './pretrain_models/HumanML3D/VQVAE/net_last.pth'
args.resume_trans = './keep_net/80-81/net_best_fid.pth'
args.down_t = 2
args.depth = 3
args.block_size = 51
import clip
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='/data/zhongchongyang/motiondiffuse')  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

trans_encoder = trans.Text2Motion_Cross_Transformer(num_vq=args.nb_code,
                                              embed_dim=1024,
                                              clip_dim=args.clip_dim,
                                              block_size=args.block_size,
                                              num_layers=9,
                                              n_head=16,
                                              drop_out_rate=args.drop_out_rate,
                                              fc_rate=args.ff_rate,
                                              num_layers_cross=args.num_layers_cross,
                                              alpha=0.5)

print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu', weights_only=True)
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

print ('loading transformer checkpoint from {}'.format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location='cpu', weights_only=True)
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval()
trans_encoder.cuda()

mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
input_text = "a person walks up to a backwards chair and sits down on it with legs outstretched, then stands back up"
output_type = "smpl"
outdir = './smpl_viz_new/0.5/'

clip_text = ["a person is jumping to the left"] 
clip_text.append("a person is running to the left.") #12
clip_text.append("a person does breakdance, and then makes a backflip.")

clip_text.append("the man is doing ballet, twirls then into a pose.")  #10498
clip_text.append("a man is tossing a coin and catching it.") #5

clip_text.append("The person sits down and breathes in a Yoga style.") #10
clip_text.append("a person flips clockwise from their left side of body and also swings both hands to their left side during flip") #11



#clip_text.append("The person is lying in bed and reach out to an alarm for 5 more minutes.") #3
#clip_text.append("The person is attempting to sit on a chair while getting drunk.") #6
#clip_text.append("The person text on a phone held sideways, and his thumbs is interacting with the screen.") #7
#clip_text.append("he person is entering a room, looking tired, slightly hunched over, head tilted down.")#8

key_points = []
target_list = [ "000002", "000012", "006211", "010498", "000000","000010", "004709"] 
for name in target_list:
    with open(pjoin("./eval_joints", name + '.json'))as f:
        key_point = json.load(f)
        for i, k in enumerate(key_point):
            if(np.shape(k) != (17,2)):
                key_point[i] = np.zeros((17,2))
        key_point = np.array(key_point)
        k_shape = key_point.shape
        key_point = key_point.reshape(k_shape[0], k_shape[1] * k_shape[2]) #[frame, joints* 2 (34)]
        if(k_shape[0]<180):
                key_point = np.concatenate([key_point, np.zeros(((180 - k_shape[0]), k_shape[1] * k_shape[2]))], axis=0)
        else:
            key_point = key_point[:180]
        key_points.append(key_point)
for j in range(len(clip_text)):
    text = clip.tokenize(clip_text[j], truncate=True).cuda()
    word_emb = clip_model.token_embedding(text).type(clip_model.dtype)
    word_emb = word_emb + clip_model.positional_embedding.type(clip_model.dtype)
    word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
    word_emb = clip_model.transformer(word_emb)
    word_emb = clip_model.ln_final(word_emb).permute(1, 0, 2).float()
    feat_clip_text = clip_model.encode_text(text).float()
    key = torch.tensor([key_points[j]]).float().cuda()
    from utils.motion_process import recover_from_ric
    import visualization.plot_3d_global as plot_3d
    from render_final import render
    for i in range(3):
        index_motion = trans_encoder.sample(feat_clip_text[0:1],word_emb[0:1], key, True)
        pred_pose = net.forward_decoder(index_motion)

        pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)
        pred_xyz = pred_xyz.squeeze(0).detach().cpu().numpy()
        if output_type =="smpl":
            print('pred', pred_xyz.shape, 'visualizing:'+clip_text[j])
            if not os.path.exists(outdir + clip_text[j]):
                os.makedirs(outdir + clip_text[j])
            render(pred_xyz, outdir=outdir + clip_text[j]+'/', device_id=0, name=str(i), pred=True)
        elif output_type =="skeleton":
            pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(),clip_text, [target_list[j] + "_" +str(i)+'.gif'])
        else:
            print("Wrong output type")
