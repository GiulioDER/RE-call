-- recall:transactional
-- Provenance for a calibration whose threshold was INHERITED from an earlier generation rather
-- than fitted on this one.
--
-- Nullable and additive on purpose. Every artifact written before this migration has
-- `carry_forward IS NULL`, which is the true statement about it: its threshold was fitted on its
-- own generation. Backfilling a default would assert provenance nobody measured.
--
-- The column is also inside the artifact checksum (see `CalibrationArtifactV2.immutable_payload`,
-- which adds the key only when it is not null, so existing checksums still verify). That is the
-- point: an unchecksummed provenance field could be edited to hide that a threshold was carried
-- rather than measured, and "this number was inherited" is exactly the claim an operator must be
-- able to trust. The payload-immutability trigger from 0011 already blocks UPDATE; the checksum
-- is the half of that guarantee which survives a direct write by a superuser.
ALTER TABLE recall_calibrations ADD COLUMN IF NOT EXISTS carry_forward JSONB;

-- A carried-forward artifact must name the parent it inherited from and the corpus delta it was
-- allowed to cross. Without both, the row records that something was inherited while refusing to
-- say from where or how far, which is worse than not recording it.
ALTER TABLE recall_calibrations DROP CONSTRAINT IF EXISTS recall_calibrations_carry_forward_shape;
ALTER TABLE recall_calibrations ADD CONSTRAINT recall_calibrations_carry_forward_shape CHECK (
    carry_forward IS NULL
    OR (
        carry_forward ? 'parent_calibration_id'
        AND carry_forward ? 'parent_generation_id'
        AND carry_forward ? 'corpus_delta'
        AND jsonb_typeof(carry_forward -> 'corpus_delta') = 'number'
        AND (carry_forward ->> 'corpus_delta')::double precision BETWEEN 0.0 AND 1.0
    )
);

-- Answers "which thresholds in this tenant are inherited rather than measured", which is the
-- question an operator asks after a chain of rebuilds and cannot answer from the artifact list.
CREATE INDEX IF NOT EXISTS recall_calibrations_carried_forward_idx
    ON recall_calibrations (tenant_id, generation_id)
    WHERE carry_forward IS NOT NULL;
