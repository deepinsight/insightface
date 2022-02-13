import torch

@torch.no_grad()
def concat_all_gather(tensor):
	"""
	Performs all_gather operation on the provided tensors.
	*** Warning ***: torch.distributed.all_gather has no gradient.
	"""
	tensors_gather = [torch.ones_like(tensor)
		for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

	output = torch.cat(tensors_gather, dim=0)
	return output

@torch.no_grad()
def batch_shuffle_ddp(x, rank, world_size):
	"""
	Batch shuffle, for making use of BatchNorm.
	*** Only support DistributedDataParallel (DDP) model. ***
	"""
	# gather from all gpus
	batch_size_this = x.shape[0]
	x_gather = concat_all_gather(x)
	batch_size_all = x_gather.shape[0]


	# random shuffle index
	idx_shuffle = torch.randperm(batch_size_all).cuda()

	# broadcast to all gpus
	torch.distributed.broadcast(idx_shuffle, src=0)

	# index for restoring
	idx_unshuffle = torch.argsort(idx_shuffle)

	# shuffled index for this gpu
	idx_this = idx_shuffle.view(world_size, -1)[rank]

	return x_gather[idx_this], idx_unshuffle

@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle, rank, world_size):
	"""
	Undo batch shuffle.
	*** Only support DistributedDataParallel (DDP) model. ***
	"""
	# gather from all gpus
	batch_size_this = x.shape[0]
	x_gather = concat_all_gather(x)
	batch_size_all = x_gather.shape[0]


	# restored index for this gpu
	idx_this = idx_unshuffle.view(world_size, -1)[rank]

	return x_gather[idx_this]
