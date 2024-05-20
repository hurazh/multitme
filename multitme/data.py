import numpy as np
import pandas as pd
from tqdm import tqdm
from readimc import MCDFile
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from PIL import Image
import json
import os
import multitme.utils as utils
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class IMC_data:
    def __init__(self, imc_file, nucleus_seg_path=None, segmentation_channels=None, data_path='./'):
        self.imc_file = imc_file
        # prepare output folder
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.data_path = data_path
        
        self.imc_acqs, self.imc_data = self.read_imc_data()
        self.nucleus_seg_path = nucleus_seg_path
        if self.nucleus_seg_path is None:
            import deepcell.applications
            self.deepcell = deepcell.applications.NuclearSegmentation()
            self.channels_to_use = segmentation_channels
        
    def read_imc_data(self):
        imc_acqs = []
        imc_data = []
        with MCDFile(self.imc_file) as f:
            slide = f.slides[0]
            self.panorama = [
                x for x in slide.panoramas if x.metadata["Type"] == "Instrument"
            ][0]
            panorama_img = Image.fromarray(f.read_panorama(self.panorama)).convert("RGBA")
            if not os.path.exists(os.path.join(self.data_path, 'imc_meta')):
                os.makedirs(os.path.join(self.data_path, 'imc_meta'))
            panorama_img.save(os.path.join(self.data_path, 'imc_meta/IMC_img.png'))
            self.panorama_img_size = panorama_img.size
            self.channel_names = np.array([
                (a if a else b)
                for (a, b) in zip(slide.acquisitions[0].channel_labels, slide.acquisitions[0].channel_names)
            ])
            for acquisition_idx, acquisition in tqdm(enumerate(slide.acquisitions), desc='Load IMC'):
                # acquisition meta data
                imc_acqs.append(acquisition)
                # acquisition reads
                imc_data.append(f.read_acquisition(acquisition))
        self.n_acq = len(imc_acqs)
        return imc_acqs, imc_data
    
    def nucleus_segmentation(self, acq):
        if self.nucleus_seg_path:
            deepcell_seg = '{}_{}.npy'.format(self.nucleus_seg_path, acq)
            return np.load(deepcell_seg)
        else:
            if len(self.channels_to_use) == 1:
                im = utils.remove_outliers(
                    utils.extract_channel(
                        self.imc_acqs[acq], self.imc_data[acq], self.channels_to_use[0]
                    )
                )[np.newaxis, ..., np.newaxis]
            elif len(self.channels_to_use) > 1:
                im = utils.remove_outliers(
                    utils.extract_maximum_projection_of_channels(
                        self.imc_acqs[acq], self.imc_data[acq], self.channels_to_use
                    )
                )[np.newaxis, ..., np.newaxis]

            labeled_nuclear_arr = self.deepcell.predict(im, image_mpp=1.0)
            np.save(os.path.join(self.data_path, 'imc_meta/cells_mask_deepcell_full_acq{}.npy'.format(acq)), 
                    labeled_nuclear_arr[0, :, :, 0])
            return labeled_nuclear_arr[0, :, :, 0]
            
    def get_cell_centers_sparse(self, data):
        cols = np.arange(data.size)
        M = csr_matrix((cols, (np.ravel(data), cols)), shape=(data.max() + 1, data.size))
        return np.array([np.mean(np.unravel_index(row.data, data.shape), 1) for R, row in enumerate(M) 
                         if ((R > 0) and (row.data.shape[0] > 0))])

    def get_tissue_map(self, pixel_sum, pixel_sum_threshold):
        exp_ind = (pixel_sum > pixel_sum_threshold).astype(int)
        exp_ind_d = np.zeros_like(exp_ind)
        n_neighbours = np.zeros_like(exp_ind)
        exp_ind_d[1:, :] += exp_ind[:-1, :]
        exp_ind_d[:-1, :] += exp_ind[1:, :]
        exp_ind_d[:, 1:] += exp_ind[:, :-1]
        exp_ind_d[:, :-1] += exp_ind[:, 1:]
        n_neighbours[1:, :] += 1
        n_neighbours[:-1, :] += 1
        n_neighbours[:, 1:] += 1
        n_neighbours[:, :-1] += 1
        exp_ind_d = exp_ind_d / n_neighbours
        tissue_map = (exp_ind_d > 0) & (exp_ind > 0)
        return tissue_map

    def get_imc_count_acq(self, acq, pixel_sum_threshold, protein_filter, return_detail=False):
        imc_data_acq = self.imc_data[acq].reshape((50, -1)).T
        cell_mask = self.nucleus_segmentation(acq)
        cell_centers = self.get_cell_centers_sparse(cell_mask)
        xv, yv = np.mgrid[:cell_mask.shape[0], :cell_mask.shape[1]]
        coords = np.array([xv.reshape(-1), yv.reshape(-1)]).T
        tr = cKDTree(cell_centers)
        dist, dist_idx = tr.query(coords, k=1)
        segmentation = (dist_idx + 1).reshape(cell_mask.shape)
        
        pixel_cliped = imc_data_acq[:, protein_filter]
        n_proteins = protein_filter.sum()
        for k in range(n_proteins):
            pixel_cliped[:, k] = pixel_cliped[:, k].clip(0, np.percentile(pixel_cliped[:, k], 99))
        
        pixel_sum = pixel_cliped.sum(axis=1).reshape(cell_mask.shape)
        tissue_map = self.get_tissue_map(pixel_sum, pixel_sum_threshold)
        imc_cell_mask = np.zeros_like(pixel_sum).astype(int)
        imc_cell_mask[tissue_map] = segmentation[tissue_map]
        
        cells = np.unique(imc_cell_mask)[1:]
        cols = np.arange(imc_cell_mask.size)
        M = csr_matrix((cols, (np.ravel(imc_cell_mask), cols)), shape=(imc_cell_mask.max() + 1, imc_cell_mask.size))
        imc_count = np.array([np.array([pixel_cliped[row.data].sum(axis=0), pixel_cliped[row.data].mean(axis=0)]) 
                 for R, row in enumerate(M) if ((R > 0) and (row.data.shape[0] > 0))])
        if return_detail:
            return imc_count[:, 0], imc_count[:, 1], pixel_cliped, segmentation, tissue_map
        else:
            return imc_count[:, 0], imc_count[:, 1]
    
    def process_data(self, pixel_sum_threshold, selected_protein):
        protein_filter = np.array([p in selected_protein for p in self.channel_names])
        for acq_idx in range(self.n_acq):
            logger.info('Cell segmentation acquisition {}'.format(acq_idx))
            pixsum, pixmean = self.get_imc_count_acq(acq_idx, pixel_sum_threshold, protein_filter)
            if not os.path.exists(os.path.join(self.data_path, 'imc_reads')):
                os.makedirs(os.path.join(self.data_path, 'imc_reads'))
            np.save(os.path.join(self.data_path, 'imc_reads/imc_count_sum_acq{}.npy'.format(acq_idx)), pixsum)
            np.save(os.path.join(self.data_path, 'imc_reads/imc_count_mean_acq{}.npy'.format(acq_idx)), pixmean)
            np.save(os.path.join(self.data_path, 'imc_meta/IMC_biomarkers.npy'), self.channel_names[protein_filter])
        self.get_offsets()

    def get_offsets(self):
        max_x = max(
            [
                float(self.panorama.metadata["SlideX1PosUm"]),
                float(self.panorama.metadata["SlideX2PosUm"]),
                float(self.panorama.metadata["SlideX3PosUm"]),
                float(self.panorama.metadata["SlideX4PosUm"]),
            ]
        )

        min_x = min(
            [
                float(self.panorama.metadata["SlideX1PosUm"]),
                float(self.panorama.metadata["SlideX2PosUm"]),
                float(self.panorama.metadata["SlideX3PosUm"]),
                float(self.panorama.metadata["SlideX4PosUm"]),
            ]
        )

        max_y = max(
            [
                float(self.panorama.metadata["SlideY1PosUm"]),
                float(self.panorama.metadata["SlideY2PosUm"]),
                float(self.panorama.metadata["SlideY3PosUm"]),
                float(self.panorama.metadata["SlideY4PosUm"]),
            ]
        )

        min_y = min(
            [
                float(self.panorama.metadata["SlideY1PosUm"]),
                float(self.panorama.metadata["SlideY2PosUm"]),
                float(self.panorama.metadata["SlideY3PosUm"]),
                float(self.panorama.metadata["SlideY4PosUm"]),
            ]
        )

        # convert x pixels to um
        x_um_per_pixel = self.panorama_img_size[0] / (max_x - min_x)
        y_um_per_pixel = self.panorama_img_size[1] / (max_y - min_y)
        
        x_pixel = []
        y_pixel = []
        for acquisition in self.imc_acqs:
            x1 = float(acquisition.metadata["ROIStartXPosUm"]) / 1000.0
            y1 = float(acquisition.metadata["ROIStartYPosUm"]) / 1000.0
            x2 = float(acquisition.metadata["ROIEndXPosUm"])
            y2 = float(acquisition.metadata["ROIEndYPosUm"])

            x_min_acq = min(x1, x2)
            x_max_acq = max(x1, x2)
            y_min_acq = min(y1, y2)
            y_max_acq = max(y1, y2)
            x_pixel.append(int((x_min_acq - min_x) / x_um_per_pixel)) 
            y_pixel.append(int((max_y - y_max_acq) / y_um_per_pixel))
        np.save(os.path.join(self.data_path, 'imc_meta/roi_offsets_x.npy'), np.array(x_pixel))
        np.save(os.path.join(self.data_path, 'imc_meta/roi_offsets_y.npy'), np.array(y_pixel))



class IMC_alignment:
    def __init__(self, data_path):
        self.data_path = data_path
        self.imc_cyto_json = os.path.join(self.data_path, 'IMC_cytassist_alignment.json')
        self.st_cyto_json = os.path.join(self.data_path, 'ST_cytassist_alignment.json')
        self.scalefactors = os.path.join(self.data_path, 'scalefactors_json.json')
        self.st_tissue_positions = os.path.join(self.data_path, 'tissue_positions.csv')
        self.st_barcodes = os.path.join(self.data_path, 'barcodes_filtered.tsv.gz')
        self.M_cyto_st = self.calculate_M_cyto_st()
        self.tissue_hires_scalef, self.ST_highres_r, self.spot_pos = self.calculate_spot_positions()

    def calculate_M_cyto_st(self):
        with open(self.imc_cyto_json) as f:
            imc_data = json.load(f)
        M_imc_cyto = np.array(imc_data['cytAssistInfo']['transformImages'])
        with open(self.st_cyto_json) as f:
            st_data = json.load(f)
        M_st_cyto = np.array(st_data['cytAssistInfo']['transformImages'])
        M_cyto_st = np.linalg.inv(M_st_cyto).dot(M_imc_cyto)
        return M_cyto_st

    def calculate_spot_positions(self):
        with open(self.scalefactors) as f:
            d = json.load(f)
        tissue_hires_scalef = d['tissue_hires_scalef']
        ST_fullres_r = d['spot_diameter_fullres']
        ST_highres_r = ST_fullres_r * tissue_hires_scalef / 2

        pos = pd.read_csv(self.st_tissue_positions)
        in_tissue_spots = pos[pos['in_tissue'] == 1]
        spots_y = in_tissue_spots['pxl_row_in_fullres'] * tissue_hires_scalef
        spots_x = in_tissue_spots['pxl_col_in_fullres'] * tissue_hires_scalef

        spot_idx_st = self.align_imc_ST()
        spot_pos = np.zeros((2, spot_idx_st.shape[0]))
        spot_pos[:, spot_idx_st] = np.array([spots_x, spots_y])
        return tissue_hires_scalef, ST_highres_r, spot_pos

    @staticmethod
    def get_cell_centers_sparse(data):
        cols = np.arange(data.size)
        M = csr_matrix((cols, (np.ravel(data), cols)), shape=(data.max() + 1, data.size))
        return np.array([np.mean(np.unravel_index(row.data, data.shape), 1) for R, row in enumerate(M) 
                         if ((R > 0) and (row.data.shape[0] > 0))])[:, ::-1]

    @staticmethod
    def transform_cells(cells_pos, offset_x, offset_y, M):
        '''Transform cell pos based on transformation matrix M'''
        n_cells = cells_pos.shape[0]
        pos_pad = np.stack((cells_pos[:, 0] + offset_x, cells_pos[:, 1] + offset_y, np.ones(n_cells)))
        return M.dot(pos_pad).T

    def map_imc_to_ST(self, imc_cell_mask, x, y):
        cells_pos = self.get_cell_centers_sparse(imc_cell_mask)
        return self.transform_cells(cells_pos, x, y, self.M_cyto_st)

    def align_imc_ST(self):
        # spot_imc = np.zeros(self.spot_pos.shape[0]).astype(bool)
        # tree = cKDTree(cell_ST[:, :2])
        # for i, spot in enumerate(self.spot_pos):
        #     dist, _ = tree.query(spot)
        #     if dist < self.ST_highres_r * 1.5:
        #         spot_imc[i] = True

        spot_barcodes = pd.read_csv(self.st_barcodes, header=None).values
        in_tissue_spots = pd.read_csv(self.st_tissue_positions)
        in_tissue_spots = in_tissue_spots[in_tissue_spots['in_tissue'] == 1]
        spot_imc_barcodes = in_tissue_spots['barcode'].values
        spot_idx_st = np.array([np.argwhere(spot_barcodes == barcode)[0][0] for barcode in spot_imc_barcodes])

        return spot_idx_st

    @staticmethod
    def imc_st_edges(cell_ST, spot_pos):
        edges = []
        edge_counter = np.zeros(cell_ST.shape[0])
        for i in range(cell_ST.shape[0]):
            spot_dists = np.sqrt(((cell_ST[i, :2] - spot_pos.T) ** 2).sum(axis=1))
            neighbours = spot_dists.argsort()[:5]
            for n in neighbours:
                if spot_dists[n] < 30:
                    edges.append([i, n])
                    edge_counter[i] += 1
        return np.array(edges), edge_counter > 0

    def process_acquisition(self, bayesTME_prop):
        x = np.load(os.path.join(self.data_path, 'imc_meta/roi_offsets_x.npy'))
        y = np.load(os.path.join(self.data_path, 'imc_meta/roi_offsets_y.npy'))
        num_acquisitions = len(x)
        cell_masks = [np.load(os.path.join(self.data_path, 'imc_meta/cells_mask_deepcell_full_acq{}.npy'.format(i))) 
                     for i in range(num_acquisitions)]
        cell_ST = []
        cell_filter = []
        edges = []

        for i in range(num_acquisitions):
            cell_ST_acq = self.map_imc_to_ST(cell_masks[i], x[i], y[i])
            cell_ST.append(cell_ST_acq)
            _, cell_filter_acq = self.imc_st_edges(cell_ST_acq, self.spot_pos)
            cell_filter.append(cell_filter_acq)
            edges_acq, cf = self.imc_st_edges(cell_ST_acq[cell_filter_acq], self.spot_pos)
            edges.append(edges_acq)

        all_imc_cells = np.concatenate([cell_ST[i][cell_filter[i]][:, :2] for i in range(num_acquisitions)])
        np.save(os.path.join(self.data_path, 'imc_meta/all_imc_cells.npy'), all_imc_cells)
        
        all_edges = []
        cells_counter = 0
        for i in range(num_acquisitions):
            edges_reindex = edges[i].copy()
            edges_reindex[:, 0] += cells_counter
            np.save(os.path.join(self.data_path, 'imc_reads/imc_st_edge_acq{}.npy'.format(i)), edges_reindex)
            all_edges.append(edges_reindex)
            cells_counter = edges_reindex[:, 0].max() + 1
        all_edges = np.concatenate(all_edges)

        ST_spatial_ref = np.zeros((all_imc_cells.shape[0], 22))
        for i in range(all_imc_cells.shape[0]):
            st_idx = all_edges[all_edges[:, 0] == i][:, 1]
            ST_spatial_ref[i] = bayesTME_prop[st_idx].mean(axis=0)
        np.save(os.path.join(self.data_path, 'imc_reads/ST_spatial_reference.npy'), ST_spatial_ref)
        
        Obs_imc = np.concatenate([np.load(
            os.path.join(self.data_path, 'imc_reads/imc_count_sum_acq{}.npy'.format(i)))[cell_filter[i]] 
                                  for i in range(num_acquisitions)], dtype='float32')
        np.save(os.path.join(self.data_path, 'imc_reads/Obs_imc.npy'), Obs_imc)
        
        return cell_ST, all_edges