def poly_lr_scheduler(optimizer, init_lr, iter, max_iter=300, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param max_iter is number of maximum iterations
		:param power is a polymomial power
	"""
	lr = init_lr * (1 - iter / max_iter) ** power
	optimizer.param_groups[0]['lr'] = lr
	
	return lr