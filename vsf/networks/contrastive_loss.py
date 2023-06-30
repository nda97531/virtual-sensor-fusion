import torch as tr


def cocoa2_loss(features_dict: dict, temp: float):
    """
    Calculate contrastive loss from
    Args:
        features_dict: dict[modal name] = Tensor shape [batch, feature]

    Returns:
        a pytorch float, contrastive loss between modalities
    """
    assert len(features_dict) > 1, 'At least 2 modals are required for contrastive loss'
    all_features = tr.stack([modal_tensor for modal_tensor in features_dict.values()])
    num_modal, batch_size, n_channel = all_features.shape

    # positive pairs
    sim = tr.matmul(all_features.permute([1, 0, 2]), all_features.permute([1, 2, 0]))
    sim = 1 - sim
    sim = tr.exp(sim / temp)
    # [batch size,]; each is sum of distances between all possible pair of modals in a batch index
    pos_error = tr.mean(sim, dim=(1, 2))

    # negative pairs
    neg_error = 0
    for i in range(num_modal):
        sim = tr.matmul(all_features[i], all_features[i].permute([1, 0]))
        sim = tr.exp(sim / temp)
        tri_mask = tr.full([batch_size, batch_size], fill_value=True)
        tri_mask[tr.arange(batch_size), tr.arange(batch_size)] = False
        off_diag_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [batch_size, batch_size - 1])
        neg_error += (tf.reduce_mean(off_diag_sim, axis=-1))


def loss(ytrue, ypred):
    # ypred [modal, batch, channel]
    batch_size, dim_size = ypred.shape[1], ypred.shape[0]
    # Positive Pairs
    pos_error = []
    for i in range(batch_size):
        sim = tf.linalg.matmul(ypred[:, i, :], ypred[:, i, :], transpose_b=True)
        sim = tf.subtract(tf.ones([dim_size, dim_size], dtype=tf.dtypes.float32), sim)
        sim = tf.exp(sim / self.temperature)
        pos_error.append(tf.reduce_mean(sim))
    # Negative pairs
    neg_error = 0
    for i in range(dim_size):
        sim = tf.cast(tf.linalg.matmul(ypred[i], ypred[i], transpose_b=True), dtype=tf.dtypes.float32)
        sim = tf.exp(sim / self.temperature)
        tri_mask = np.ones(batch_size ** 2, dtype=np.bool).reshape(batch_size, batch_size)
        tri_mask[np.diag_indices(batch_size)] = False
        off_diag_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [batch_size, batch_size - 1])
        neg_error += (tf.reduce_mean(off_diag_sim, axis=-1))

    error = tf.multiply(tf.reduce_sum(pos_error), self.scale_loss) + self.lambd * tf.reduce_sum(neg_error)

    return error


if __name__ == '__main__':
    features = {
        'acc': tr.rand([4, 15]),
        'gyro': tr.rand([4, 15]),
        'skeleton': tr.rand([4, 15]),
    }
    cocoa2_loss(features, temp=1)


    def num_dist(a, b):
        num = a * (a - 1) / 2
        m_dist = b ** 2 - b
        return num * m_dist


    print(num_dist(3, 4))
    print(num_dist(4, 3))
