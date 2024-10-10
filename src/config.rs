use std::path::PathBuf;

use burn::prelude::*;

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "None")]
    sys_home_dir: Option<PathBuf>,
    #[config(default = "PathBuf::from(\"datasets/dis\")")]
    data_root_dir: PathBuf,
    #[config(default = "Task::DIS5K")]
    task: Task,
    // training_set: String,
    #[config(default = "Prompt4loc::Dense")]
    prompt4_loc: Prompt4loc,
    #[config(default = "true")]
    pub(crate) ms_supervision: bool,
    #[config(default = "true")]
    pub(crate) out_ref: bool,
    #[config(default = "true")]
    pub(crate) dec_ipt: bool,
    #[config(default = "true")]
    pub(crate) dec_ipt_split: bool,
    #[config(default = "3")]
    cxt_num: usize,
    #[config(default = "MulSclIpt::Cat")]
    pub(crate) mul_scl_ipt: MulSclIpt,
    #[config(default = "DecAtt::ASPPDeformable")]
    pub(crate) dec_att: DecAtt,
    #[config(default = "SqueezeBlock::BasicDecBlk(1)")]
    pub(crate) squeeze_block: SqueezeBlock,
    #[config(default = "DecBlk::BasicDecBlk")]
    pub(crate) dec_blk: DecBlk,
    // batch_size: usize,
    // finetune_last_epochs: usize,
    // lr: f64,
    // size: Vec<usize>,
    // num_workers: usize,
    #[config(default = "Backbone::SwinV1L")]
    pub(crate) backbone: Backbone,
    #[config(default = "LatBlk::BasicLatBlk")]
    pub(crate) lat_blk: LatBlk,
    #[config(default = "DecChannelsInter::Fixed")]
    pub(crate) dec_channels_inter: DecChannelsInter,
    #[config(default = "Refine::None")]
    refine: Refine,
    // progressive_ref: String,
    // ender: String,
    // scale: String,
    // auxiliary_classification: bool,
    // refine_iteration: usize,
    // freeze_bb: bool,
    // preproc_methods: Vec<String>,
    // optimizer: String,
    // lr_decay_epochs: Vec<usize>,
    // lr_decay_rate: f64,
    // lambdas_pix_last: HashMap<String, f64>,
    // lambdas_cls: LambdasCls,
    // lambda_adv_g: usize,
    // lambda_adv_d: usize,
    // weights_root_dir: PathBuf,
    // weights: PathBuf,
    // batch_size_valid: usize,
    // rand_seed: usize,
    // save_last: usize,
    // save_step: usize,
}

impl ModelConfig {
    pub fn lateral_channels_in_collection(&self) -> [usize; 4] {
        let channels = match self.backbone {
            Backbone::Vgg16 | Backbone::Vgg16bn => [512, 256, 128, 64],
            Backbone::Resnet50 => [1024, 512, 256, 64],
            Backbone::SwinV1T | Backbone::SwinV1S => [768, 384, 192, 96],
            Backbone::SwinV1B => [1024, 512, 256, 128],
            Backbone::SwinV1L => [1536, 768, 384, 192],
            Backbone::PvtV2B0 => [256, 160, 64, 32],
            Backbone::PvtV2B1 | Backbone::PvtV2B2 | Backbone::PvtV2B5 => [512, 320, 128, 64],
        };
        if self.mul_scl_ipt == MulSclIpt::Cat {
            let [c1, c2, c3, c4] = channels;
            [c1 * 2, c2 * 2, c3 * 2, c4 * 2]
        } else {
            channels
        }
    }

    pub fn cxt(&self) -> [usize; 3] {
        if self.cxt_num > 0 {
            let reversed: Vec<usize> = self.lateral_channels_in_collection()[1..]
                .iter()
                .rev()
                .cloned()
                .collect();
            reversed[reversed.len().saturating_sub(self.cxt_num)..]
                .try_into()
                .unwrap()
        } else {
            [0, 0, 0]
        }
    }
}

#[derive(Config, Debug)]
pub enum Task {
    DIS5K,
    COD,
    HRSOD,
    General,
    General2k,
    Matting,
}

#[derive(Config, Debug)]
pub enum Prompt4loc {
    Dense,
    Sparse,
}

#[derive(Config, Debug, PartialEq)]
pub enum MulSclIpt {
    None,
    Add,
    Cat,
}

#[derive(Config, Debug)]
pub enum DecAtt {
    None,
    ASPP,
    ASPPDeformable,
}

#[derive(Config, Debug, PartialEq)]
pub enum SqueezeBlock {
    None,
    BasicDecBlk(usize),
    ResBlk(usize),
    ASPP(usize),
    ASPPDeformable(usize),
}

impl SqueezeBlock {
    pub fn count(&self) -> usize {
        match self {
            SqueezeBlock::None => 0,
            SqueezeBlock::BasicDecBlk(x) => *x,
            SqueezeBlock::ResBlk(x) => *x,
            SqueezeBlock::ASPP(x) => *x,
            SqueezeBlock::ASPPDeformable(x) => *x,
        }
    }
}

#[derive(Config, Debug)]
pub enum DecBlk {
    BasicDecBlk,
    ResBlk,
}

#[derive(Config, Debug)]
pub enum Backbone {
    Vgg16,
    Vgg16bn,
    Resnet50, // 0, 1, 2
    SwinV1T,
    SwinV1S, // 3, 4
    SwinV1B,
    SwinV1L, // 5-bs9, 6-bs4
    PvtV2B0,
    PvtV2B1, // 7, 8
    PvtV2B2,
    PvtV2B5, // 9-bs10, 10-bs5
}

#[derive(Config, Debug)]
pub enum LatBlk {
    BasicLatBlk,
}

#[derive(Config, Debug)]
pub enum DecChannelsInter {
    Fixed,
    Adap,
}

#[derive(Config, Debug)]
pub enum Refine {
    None,
    Itself,
    RefUNet,
    Refiner,
    RefinerPVTInChannels4,
}

#[derive(Config, Debug)]
pub enum PreprocMethods {
    Flip,
    Enhance,
    Rotate,
    Pepper,
    Crop,
}

#[derive(Config, Debug)]
pub enum Optimizer {
    Adam,
    AdamW,
}
