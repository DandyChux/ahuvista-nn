-- BigQuery extraction for admission-level maternal mortality training data.
-- The resulting CSV is consumed with the mapping in manifest.example.json.
-- This query deliberately uses an admission-level patient_id so separate
-- hospital admissions are not merged during ingestion.

WITH maternal_admissions AS (
    SELECT DISTINCT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.race,
        a.hospital_expire_flag AS mortality_label
    FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
        ON a.subject_id = p.subject_id
    LEFT JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` diag
        ON a.hadm_id = diag.hadm_id
    WHERE p.gender = 'F'
      AND p.anchor_age >= 18
      AND (
          STARTS_WITH(diag.icd_code, '63') OR
          STARTS_WITH(diag.icd_code, '64') OR
          STARTS_WITH(diag.icd_code, '65') OR
          STARTS_WITH(diag.icd_code, '66') OR
          STARTS_WITH(diag.icd_code, '67') OR
          STARTS_WITH(diag.icd_code, 'V27') OR
          STARTS_WITH(diag.icd_code, 'O') OR
          STARTS_WITH(diag.icd_code, 'Z37')
      )
),

first_icu_stay AS (
    SELECT
        hadm_id,
        stay_id,
        intime,
        outtime,
        ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime ASC) AS rn
    FROM `physionet-data.mimiciv_3_1_icu.icustays`
),

vitals_first_24h AS (
    SELECT
        ie.hadm_id,
        AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) AS avg_heart_rate,
        MAX(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) AS max_heart_rate,
        AVG(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) AS avg_sys_bp,
        MAX(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) AS max_sys_bp,
        AVG(CASE WHEN ce.itemid IN (220051, 220180) THEN ce.valuenum END) AS avg_dia_bp,
        MAX(CASE WHEN ce.itemid IN (220051, 220180) THEN ce.valuenum END) AS max_dia_bp,
        MIN(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END) AS min_spo2
    FROM first_icu_stay ie
    JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ce
        ON ie.stay_id = ce.stay_id
    WHERE ie.rn = 1
      AND ce.valuenum IS NOT NULL
      AND ce.charttime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL 24 HOUR)
      AND (
          (ce.itemid = 220045 AND ce.valuenum BETWEEN 30 AND 220) OR
          (ce.itemid IN (220050, 220179) AND ce.valuenum BETWEEN 50 AND 250) OR
          (ce.itemid IN (220051, 220180) AND ce.valuenum BETWEEN 20 AND 150) OR
          (ce.itemid = 220277 AND ce.valuenum BETWEEN 50 AND 100)
      )
    GROUP BY ie.hadm_id
),

labs_first_24h AS (
    SELECT
        ma.hadm_id,
        MIN(CASE WHEN le.itemid = 51221 THEN le.valuenum END) AS min_hematocrit,
        MIN(CASE WHEN le.itemid = 51222 THEN le.valuenum END) AS min_hemoglobin,
        MIN(CASE WHEN le.itemid = 51265 THEN le.valuenum END) AS min_platelets,
        MAX(CASE WHEN le.itemid = 50912 THEN le.valuenum END) AS max_creatinine,
        MAX(CASE WHEN le.itemid = 50889 THEN le.valuenum END) AS max_ast,
        MAX(CASE WHEN le.itemid = 50883 THEN le.valuenum END) AS max_alt
    FROM maternal_admissions ma
    JOIN `physionet-data.mimiciv_3_1_hosp.labevents` le
        ON ma.hadm_id = le.hadm_id
    WHERE le.valuenum IS NOT NULL
      AND le.charttime BETWEEN ma.admittime AND DATETIME_ADD(ma.admittime, INTERVAL 24 HOUR)
      AND (
          (le.itemid = 51221 AND le.valuenum BETWEEN 5 AND 70) OR
          (le.itemid = 51222 AND le.valuenum BETWEEN 1 AND 25) OR
          (le.itemid = 51265 AND le.valuenum BETWEEN 1 AND 1500) OR
          (le.itemid = 50912 AND le.valuenum BETWEEN 0.01 AND 30) OR
          (le.itemid IN (50889, 50883) AND le.valuenum BETWEEN 0.1 AND 5000)
      )
    GROUP BY ma.hadm_id
),

discharge_notes AS (
    SELECT
        hadm_id,
        text AS discharge_summary
    FROM (
        SELECT
            hadm_id,
            text,
            ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime DESC) AS rn
        FROM `physionet-data.mimiciv_note.discharge`
    )
    WHERE rn = 1
)

SELECT
    CONCAT(CAST(ma.subject_id AS STRING), ':', CAST(ma.hadm_id AS STRING)) AS patient_id,
    ma.subject_id,
    ma.hadm_id,
    p.anchor_age,
    ma.race,
    ma.mortality_label,
    v.avg_heart_rate,
    v.max_heart_rate,
    v.avg_sys_bp,
    v.max_sys_bp,
    v.avg_dia_bp,
    v.max_dia_bp,
    v.min_spo2,
    l.min_hematocrit,
    l.min_hemoglobin,
    l.min_platelets,
    l.max_creatinine,
    l.max_ast,
    l.max_alt,
    n.discharge_summary
FROM maternal_admissions ma
JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
    ON ma.subject_id = p.subject_id
LEFT JOIN vitals_first_24h v
    ON ma.hadm_id = v.hadm_id
LEFT JOIN labs_first_24h l
    ON ma.hadm_id = l.hadm_id
LEFT JOIN discharge_notes n
    ON ma.hadm_id = n.hadm_id;
