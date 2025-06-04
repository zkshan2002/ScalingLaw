import itertools
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from seaborn import color_palette


def plot(
    png: str,
    x_list,
    y_list,
    label_list,
    x_label,
    y_label,
    vertical_line_at = None,
):

    colors = color_palette("husl", len(y_list))

    plt.rcParams["font.size"] = 12

    plt.figure(figsize=(4, 3))
    
    for i in range(len(y_list)):
        label = label_list[i] if label_list is not None else None
        plt.plot(x_list[i], y_list[i], color=colors[i], label=label, linewidth=1)

    if vertical_line_at is not None:
        plt.axvline(x=vertical_line_at, color='black', linewidth=1)

    if label_list is not None:
        plt.legend()
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    file_path = Path(png)
    if file_path.suffix != ".png":
        file_path = Path(f"{file_path.stem}.png")
    if file_path.is_file():
        file_path.unlink()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png, bbox_inches='tight', dpi=300)
    plt.close()
    return

MTL_LAMBDA = 0.999

def _huber(x, delta=1e-3):
    x_abs = np.abs(x)
    x = np.where(x_abs <= delta, 0.5 * x ** 2, delta * x_abs - 0.5 * delta ** 2)
    return x

def _evaluate(log_pred, log_gt):
    # R^2 (Coefficient of Determination)
    r2 = 1 - (np.sum((log_gt - log_pred)**2) / np.sum((log_gt - np.mean(log_pred))**2))
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(log_gt - log_pred))
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((log_gt - log_pred)**2))
    
    pred = np.exp(log_pred)
    gt = np.exp(log_gt)

    # PredE (Prediction Error) - typically the average relative error
    prede = np.mean(np.abs(gt - pred) / gt)
    
    return {
        'R2': r2.item(),
        'MAE': mae.item(),
        'RMSE': rmse.item(),
        'PredE': prede.item(),
    }

def _mtl_prepare(lr, lam):
    s1 = np.cumsum(lr)
    b = lr.shape[0]
    m = np.zeros_like(lr)
    m[0] = lr[0]
    for i in range(1, b):
        m[i] = m[i - 1] * lam + (lr[i - 1] - lr[i])
    s2 = np.cumsum(m)
    return s1, s2

def _mtl_law(params, s1, s2):
    l0, a, alpha, c = params
    pred = l0 + a * s1 ** (-alpha) - c * s2
    return pred

def _mtl_fit(loss, lr):
    def _objective_fn(params):
        # uses external s1, s2, loss
        pred = _mtl_law(params, s1, s2)
        if np.any(pred <= 0):
            return 1e10
        x = np.log(pred) - np.log(loss)
        x = _huber(x)
        return x.sum().item()
    
    s1, s2 = _mtl_prepare(lr, MTL_LAMBDA)
    l0_init_range = [1, 2, 4]
    a_init_range = [1, 2, 4]
    alpha_init_range = [0.2, 0.4, 0.6, 0.8, 1, 1.2]
    c_init_range = [0.5, 1]
    init_ranges = itertools.product(l0_init_range, a_init_range, alpha_init_range, c_init_range)
    l0_bound = (0, 10)
    a_bound = (0, 20)
    alpha_bound = (0, 10)
    c_bound = (0, 20)
    bounds = [l0_bound, a_bound, alpha_bound, c_bound]
    options = {
        'maxiter': 1_000_000,
        'ftol': 1e-9,
        'gtol': 1e-6,
        'eps': 1e-8,
    }
    best_loss = np.inf
    best_param = None
    for init_params in init_ranges:
        result = scipy.optimize.minimize(_objective_fn, init_params, method='L-BFGS-B', bounds=bounds, options=options)
        optimal_loss = result.fun
        optimal_param = result.x
        print(f'{optimal_loss:.8f} {[init_params]}->{optimal_param}')
        if optimal_loss < best_loss:
            best_loss = optimal_loss
            best_param = optimal_param
    best_param_string = ','.join(f'{v.item()}' for v in best_param)
    print(f'{best_loss}')
    print(f'{best_param_string}')
    return

def _mtl_pred(lr, params):
    s1, s2 = _mtl_prepare(lr, MTL_LAMBDA)
    pred = _mtl_law(params, s1, s2)
    return pred

def _mpl_prepare(lr):
    lr_sum = np.cumsum(lr)
    lr_gap = np.zeros_like(lr)
    lr_gap[1:] = lr[:-1] - lr[1:]
    return lr_sum, lr_gap

def _mpl_law(params, lr, lr_sum, lr_gap):
    l0, a, b, c, alpha, beta, gamma = params
    ld = np.zeros_like(lr)
    for i in range(1, len(lr)):
        ld[i] = np.sum(
            lr_gap[1:i + 1] * (1 - (c * lr[1: i + 1] ** (-gamma) * (lr_sum[i] - lr_sum[:i]) + 1) ** (-beta))
        )
    pred = l0 + a * lr_sum ** (-alpha) - b * ld * 100
    return pred

def _mpl_fit(loss, lr):
    def _objective_fn(params):
        # uses external lr, lr_sum, lr_gap, loss
        pred = _mpl_law(params, lr, lr_sum, lr_gap)
        if np.any(pred <= 0):
            return 1e10
        x = np.log(pred) - np.log(loss)
        x = _huber(x)
        return x.sum().item()
    
    lr_sum, lr_gap = _mpl_prepare(lr)
    l0_init_range = [4]
    a_init_range = [1]
    b_init_range = [2, 3, 4, 5]
    c_init_range = [2]
    alpha_init_range = [0.5]
    beta_init_range = [0.2, 0.4, 0.6, 0.8, 1.0]
    gamma_init_range = [0.5]
    init_ranges = itertools.product(l0_init_range, a_init_range, b_init_range, c_init_range, alpha_init_range, beta_init_range, gamma_init_range)
    l0_bound = (0, 10)
    a_bound = (0, 10)
    b_bound = (0, 1000)
    c_bound = (0, 10)
    alpha_bound = (0, 10)
    beta_bound = (0, 10)
    gamma_bound = (0, 10)
    bounds = [l0_bound, a_bound, b_bound, c_bound, alpha_bound, beta_bound, gamma_bound]
    options = {
        'maxiter': 1_000_000,
        'ftol': 1e-9,
        'gtol': 1e-6,
        'eps': 1e-8,
    }
    best_loss = np.inf
    best_param = None
    for init_params in init_ranges:
        result = scipy.optimize.minimize(_objective_fn, init_params, method='L-BFGS-B', bounds=bounds, options=options)
        optimal_loss = result.fun
        optimal_param = result.x
        print(f'{optimal_loss:.4f} {init_params}->{optimal_param}')
        if optimal_loss < best_loss:
            best_loss = optimal_loss
            best_param = optimal_param
    best_param_string = ','.join(f'{v.item()}' for v in best_param)
    print(f'{best_loss}')
    print(f'{best_param_string}')
    return

def _mpl_pred(lr, params):
    lr_sum, lr_gap = _mpl_prepare(lr)
    pred = _mpl_law(params, lr, lr_sum, lr_gap)
    return pred

def _load_data():
    data = pd.read_pickle(f'data/gpt_loss+lrs.pkl')
    keys = ['M:100M_gpt_D:20B_scheduler:811_rope', 'M:100M_gpt_D:20B_scheduler:wsd_rope', 'M:100M_gpt_D:20B_scheduler:cosine_rope']
    xs = []
    tags = []
    losses = []
    lrs = []
    for key in keys:
        df = data[key]
        tag = key.split(':')[-1].split('_')[0]
        tags.append(tag)

        xs.append(df['step'].to_numpy())
        losses.append(df['Metrics/loss'].to_numpy())
        lrs.append(df['lr'].to_numpy())
    return xs, losses, lrs, tags

def get_params(law, mode):
    if law == 'mtl':
        if mode == 'cosine[::100]':
            return 2.73813148159852,0.01976684625966505,0.8872279225723892,0.39947178516684223
        if mode == '811[::100]':
            return 2.8299991060655136,0.011772891763467425,0.9681195197709653,0.5499096409178682
        if mode == 'wsd[::100]':
            return 2.8136940281922627,0.012277265094319855,0.9619136746208896,0.47808484586389355
        # if mode == 'cosine[::20][:100]':
        #     return 0.3888928169231402,1.4331116854892705,0.3052849777652237,0.0
        # if mode == 'cosine[::20][:200]':
        #     return 2.1519199184523417,0.3600952479582193,0.5300834418214345,0.0
        # if mode == 'cosine[::20][:400]':
        #     return 2.4840495056021323,0.20996402372967857,0.6256738397147624,0.0
        # if mode == 'cosine[::40][:100]':
        #     return 2.208059310579839,0.2140063262143781,0.5668490743184864,0.0
    if law == 'mpl':
        if mode == 'cosine[::100]':
            return 2.7159857881686227,0.02055261518869747,4.608281601838283,1.9361977985030208,0.8816644777834804,0.5106535206882394,0.3554603986783882
        if mode == '811[::100]':
            return 2.7261517334879963,0.01952903459234997,1.4156120236732381,2.1855681473544637,0.8898864275369189,0.38471026004745623,1.163791656283293
        if mode == 'wsd[::100]':
            return 2.707721208930419,0.02129442408927263,2.957140032563492,1.9988489233057654,0.8758655320641824,0.8815419582933766,0.509273079693179
        # if mode == 'cosine[::20][:100]':
        #     return 0.40471874994080387,1.4220739003144511,4.003247274312638,2.002816909987991,0.30641131002970845,0.5185041731193332,0.538951199710121
        # if mode == 'cosine[::20][:200]':
        #     return 2.151921411851032,0.36009456813069074,2.9926438278319822,1.994903002919017,0.5300838532764911,0.0,0.0
        # if mode == 'cosine[::20][:400]':
        #     return 2.484049094057331,0.20996422979774776,0.0,1.9502859366700918,0.6256736529171578,0.5329455327762878,0.0
        # if mode == 'cosine[::40][:100]':
        #     return 2.2080881767668497,0.2139963684168267,39.831656525864645,0.0,0.5668559112380622,0.0,0.0

def fit():
    xs, losses, lrs, tags = _load_data()
    i = 0
    print(f'Fitting {tags[i]}')
    x = xs[i]
    loss = losses[i]
    lr = lrs[i]
    # downsample for efficiency: select one data point for every consecutive 100 points
    x = x[::100]
    loss = loss[::100]
    lr = lr[::100]
    # fit: will print loss and parameter to stdout. Copy them to `get_params` for evaluation.
    _mtl_fit(loss, lr)
    # _mpl_fit(loss, lr)
    return

def evaluate():
    xs, losses, lrs, tags = _load_data()
    for i in range(3):
        x = xs[i]
        loss = losses[i]
        lr = lrs[i]
        x = x[::100]
        loss = loss[::100]
        lr = lr[::100]
        params = get_params('mpl', 'wsd[::100]')
        # pred = _mtl_pred(lr, params)
        pred = _mpl_pred(lr, params)
        log_pred = np.log(pred)
        log_loss = np.log(loss)
        results = _evaluate(log_pred, log_loss)
        print({key: f'{value:.4f}' for key, value in results.items()})
    return

def visualize():
    xs, losses, lrs, tags = _load_data()
    assert tags[2] == 'cosine'
    x = xs[2]
    loss = losses[2]
    lr = lrs[2]
    x = x[::100]
    loss = loss[::100]
    lr = lr[::100]
    params = get_params('mtl', 'cosine[::100]')
    pred = _mtl_pred(lr, params)
    params = get_params('mpl', 'cosine[::100]')
    pred2 = _mpl_pred(lr, params)
    # optionally, remove initial points for better visualization
    r = 2
    x = x[r:]
    lr = lr[r:]
    pred = pred[r:]
    pred2 = pred2[r:]
    loss = loss[r:]
    # (x_list[i], y_list[i]) define a piece-wise linear curve with label label_list[i]
    plot(
        'fig/temp.png',
        # x_list=[x, x, x],
        x_list=[np.log(x), np.log(x), np.log(x)],
        # y_list=[loss, pred, pred2],
        y_list=[np.log(loss), np.log(pred), np.log(pred2)],
        label_list=['gt', 'MTL', 'MPL'],
        x_label='log steps',
        y_label='log loss',
    )


# General design:
# 1. isolate `fit`, `evaluate`, and `visualize`.
# 2. `fit` prints best parameter to stdout separated by ','. This full line can be directly copied to `get_params`.
# 3. before evaluate and visualize, register the param to use in `get_params` by assigning and passing a new `mode` string.

if __name__ == '__main__':
    fit()
    # evaluate()
