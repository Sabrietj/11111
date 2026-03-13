@REM python src/extract_feature/__label__.py
@REM python src/extract_feature/__main__.py --clean-plot-dir --force
python src/embed_feature/__main__.py

@REM python src/build_session_graph/__main__.py --sampling-ratio 1.0 --downsample_benign_only true --split-mode random --split-ratio 0.8,0.1,0.1
@REM python src/draw_session_graph/draw_session_graph.py --only_draw_non_benign --skip_labels=portscan,bot --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp
@REM python src/draw_session_graph/draw_session_graph_size_distr.py --no-cache --type both --plot-style both --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp
@REM python src/draw_session_graph/draw_label_distr.py --no-cache --type separate --split all --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp

python src/build_session_graph/__main__.py --sampling-ratio 0.02 --downsample_benign_only true --split-mode random --split-ratio 0.8,0.1,0.1
@REM python src/draw_session_graph/draw_session_graph.py --only_draw_non_benign --skip_labels=portscan,bot --graph_file_name=all_session_graph__sampled_p0.02__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp
python src/draw_session_graph/draw_session_graph_size_distr.py --no-cache --type both --plot-style both --graph_file_name=all_session_graph__sampled_p0.02__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp
python src/draw_session_graph/draw_label_distr.py --no-cache --type separate --split all --graph_file_name=all_session_graph__sampled_p0.02__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp

@REM python src/models/session_gnn_flow_bert_multiview_ssl/train.py