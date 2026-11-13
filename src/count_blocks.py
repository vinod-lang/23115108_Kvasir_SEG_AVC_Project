from model import BaselineModel, SkipFusionModel

for name, m in [("Baseline", BaselineModel()), ("Skip Fusion", SkipFusionModel())]:
    print("\n" + name)
    total = 0
    for child_name, child in m.named_children():
        if child_name == "encoder":
            for sub_name, sub in child.named_children():
                p = sum(x.numel() for x in sub.parameters())
                total += p
                print(f"  encoder.{sub_name}: {p:,}")
        else:
            p = sum(x.numel() for x in child.parameters())
            total += p
            print(f"  {child_name}: {p:,}")
    print("  TOTAL:", f"{total:,}")