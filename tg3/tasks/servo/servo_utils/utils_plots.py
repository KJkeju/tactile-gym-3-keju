import os
import numpy as np
import matplotlib.pylab as plt

from cri.transforms import quat2euler, euler2quat, inv_transform


class Contour3DPlotter:
    def __init__(self,
                 save_dir=None,
                 save_num=10,
                 name="contour_plot.png",
                 init_pose=[0, -55, 0, 0, 0, 0],
                 limits=[[-60, 60], [-60, 60], [-30, 30]],
                 elev=30,
                 azim=180+45,
                 inv='y'
        ):

        self.save_dir = save_dir
        self.save_num = save_num
        self.name = name
        self.v0 = init_pose
        self.v = np.nan*np.ones((1, 6))
        self.counter = 0
                
        plt.ion
        self._fig = plt.figure(name, figsize=(5, 5))
        self._fig.subplots_adjust(left=-0.1, right=1.1, bottom=-0.05, top=1.05)
        self._ax = self._fig.add_subplot(111, projection='3d', proj_type = 'ortho')
        self._ax.view_init(elev, azim)
        self._ax.plot(limits[0], limits[1], limits[2], ':w')
        self._ax.axis('equal')
        if inv == 'y':
            self._ax.invert_yaxis()


    def update(self, v):
        self.v = np.vstack([self.v, v - self.v0])
        self.counter += 1

        v_q = euler2quat([0, 0, 0, *self.v[-1, 3:]], axes='rxyz')
        d_q = euler2quat([0, 1, 0, 0, 0, 0], axes='rxyz')
        n_q = euler2quat([0, 0, 1, 0, 0, 0], axes='rxyz')
        p = self.v[-2:, :3]
        r = quat2euler(inv_transform(d_q, v_q), axes='rxyz')[:3]
        n = quat2euler(inv_transform(n_q, v_q), axes='rxyz')[:3]

        self._ax.plot(*p.T, '-r', lw=1)
        self._ax.plot(*(p[1,:] + np.outer(r, [-5,5]).T).T, '-b', lw=0.5)
        self._ax.plot(*(p[1,:] + np.outer(n, [0,5]).T).T, '-g', lw=0.5)

        save_now = (self.counter % self.save_num == 0)
        if save_now and self.save_dir is not None:
            save_file = os.path.join(self.save_dir, self.name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')


if __name__ == '__main__':
    pass
