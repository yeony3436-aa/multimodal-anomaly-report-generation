-- 테이블 생성
create table if not exists llava_reports (
  id bigserial primary key,

  filename text not null,
  image_path text not null,
  heatmap_path text null,
  overlay_path text null,

  dataset varchar(50) not null,
  category varchar(50) not null,
  ground_truth text null,

  decision varchar(20) not null,
  confidence double precision null,

  has_defect int not null default 0,
  defect_type text not null default '',
  location text not null default '',
  severity text not null default '',

  defect_description text not null default '',
  possible_cause text not null default '',
  product_description text not null default '',

  summary text not null default '',
  impact text not null default '',
  recommendation text not null default '',

  inference_time double precision null, -- seconds
  datetime timestamptz not null default now()
);

create index if not exists idx_llava_reports_filter
on llava_reports (dataset, category, decision, datetime desc);
