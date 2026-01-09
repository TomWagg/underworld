import numpy as np

def maxwellian(v, sigma):
    '''Maxwellian distribution'''
    return np.sqrt(2/np.pi) * v**2/sigma**3 * np.exp(-v**2/(2*sigma**2))

def kick_distribution(v, distribution='igoshev_young'):
    # Igoshev (2020) main numbers
    if distribution == 'igoshev_all':
        w = 0.42
        sigma_1 = 128
        sigma_2 = 298
    # Igoshev (2020) numbers for young pulsars
    elif distribution == 'igoshev_young':
        w = 0.20
        sigma_1 = 56
        sigma_2 = 336
    elif distribution == 'igoshev_young_half':
        sigma_2 = 336
        return maxwellian(v, sigma_2)
    elif distribution == 'igoshev_young_weighted_ECSN':
        w = 0.20
        sigma_1 = 56
        return w * maxwellian(v, sigma_1)
    elif distribution == 'renzo':
        w = 0.02
        sigma_1 = 1
        sigma_2 = 16
    elif distribution == 'hobbs':
        return maxwellian(v, 265)
    else:
        raise ValueError(f'Unknown distribution: {distribution}')

    return w * maxwellian(v, sigma_1) + (1 - w) * maxwellian(v, sigma_2)

def get_kick_distribution_uncertainties(v, distribution='igoshev_young'):
    steps = 50

    # Igoshev (2020) main numbers
    if distribution == 'igoshev_all':
        ws = [0.27, 0.42, 0.59]
        ws = np.linspace(ws[0], ws[-1], steps)
        sigma_1s = [110, 128, 150]
        sigma_1s = np.linspace(sigma_1s[0], sigma_1s[-1], steps)
        sigma_2s = [270, 298, 326]
        sigma_2s = np.linspace(sigma_2s[0], sigma_2s[-1], steps)
    # Igoshev (2020) numbers for young quasars
    elif distribution == 'igoshev_young':
        ws = [0.10, 0.20, 0.31]
        ws = np.linspace(ws[0], ws[-1], steps)
        sigma_1s = [41, 56, 81]
        sigma_1s = np.linspace(sigma_1s[0], sigma_1s[-1], steps)
        sigma_2s = [291, 336, 381]
        sigma_2s = np.linspace(sigma_2s[0], sigma_2s[-1], steps)
    else:
        raise ValueError(f'Unknown distribution: {distribution}')

    values = np.zeros((v.shape[0], steps**3))

    count = 0
    for w in ws:
        for sigma_1 in sigma_1s:
            for sigma_2 in sigma_2s:
                values[:, count] = w * maxwellian(v, sigma_1) + (1 - w) * maxwellian(v, sigma_2)
                count += 1

    return np.amin(values, axis=1), np.amax(values, axis=1)

def get_kick_weight(distribution='igoshev_young'):
    if distribution == 'igoshev_all_half':
        return 0.42
    elif distribution == 'igoshev_young_half':
        return 0.2
    else:
        raise ValueError(f'Unknown distribution half distribution: {distribution}')

def tag_kick(v, distribution='igoshev_young'):
    if distribution == 'igoshev_young':
        velocity
    else:
        raise ValueError(f'Unknown distribution distribution: {distribution}')
