#[cfg(test)]
mod tests {
    use crate::config::{Backbone, BackboneConfig, DecoderConfig, ModelConfig};
    use crate::error::BiRefNetError;

    #[test]
    fn test_unsupported_backbone_error() {
        let config =
            ModelConfig::new().with_backbone(BackboneConfig::new().with_backbone(Backbone::Vgg16));

        match config.validate() {
            Err(BiRefNetError::UnsupportedBackbone { backbone }) => {
                assert!(backbone.contains("Vgg16"));
            }
            _ => panic!("Expected UnsupportedBackbone error"),
        }
    }

    #[test]
    fn test_invalid_context_number() {
        let config = ModelConfig::new().with_decoder(DecoderConfig::new().with_cxt_num(5)); // Invalid: should be <= 3

        match config.validate() {
            Err(BiRefNetError::InvalidConfiguration { reason }) => {
                assert!(reason.contains("Context number must be <= 3"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_valid_configuration() {
        let config = ModelConfig::new()
            .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
            .with_decoder(DecoderConfig::new().with_cxt_num(3));

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_context_channels_calculation() {
        let config = ModelConfig::new()
            .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
            .with_decoder(DecoderConfig::new().with_cxt_num(3));

        let result = config.cxt();
        assert!(result.is_ok());
        let channels = result.unwrap();
        assert_eq!(channels.len(), 3);
    }

    #[test]
    fn test_context_channels_error() {
        let config = ModelConfig::new().with_decoder(DecoderConfig::new().with_cxt_num(10)); // This should cause an error in validate

        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            BiRefNetError::InvalidConfiguration { reason } => {
                assert!(reason.contains("Context number must be <= 3"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_out_ref_dependency() {
        let config = ModelConfig::new().with_decoder(
            DecoderConfig::new()
                .with_ms_supervision(false)
                .with_out_ref(true),
        ); // Invalid: out_ref=true but ms_supervision=false

        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            BiRefNetError::InvalidConfiguration { reason } => {
                assert!(reason.contains("out_ref can only be enabled when ms_supervision is true"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_dec_ipt_split_dependency() {
        let config = ModelConfig::new().with_decoder(
            DecoderConfig::new()
                .with_dec_ipt(false)
                .with_dec_ipt_split(true),
        ); // Invalid: dec_ipt_split=true but dec_ipt=false

        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            BiRefNetError::InvalidConfiguration { reason } => {
                assert!(reason.contains("dec_ipt_split can only be enabled when dec_ipt is true"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_squeeze_block_dec_att_compatibility() {
        use crate::config::{DecAtt, SqueezeBlock};

        let config = ModelConfig::new().with_decoder(
            DecoderConfig::new()
                .with_squeeze_block(SqueezeBlock::ASPP(3))
                .with_dec_att(DecAtt::None),
        ); // Invalid: squeeze_block enabled but dec_att is None

        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            BiRefNetError::InvalidConfiguration { reason } => {
                assert!(reason.contains("dec_att should not be None when squeeze_block is enabled"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_refine_not_implemented() {
        use crate::config::Refine;

        let config = ModelConfig::new()
            .with_refine(crate::config::RefineConfig::new().with_refine(Refine::RefUNet));

        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            BiRefNetError::InvalidConfiguration { reason } => {
                assert!(reason.contains("Refine is not yet implemented"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }
}
