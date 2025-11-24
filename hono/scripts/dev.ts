 import type { SpawnOptions } from "bun";

const spawnOptions: any = {
  stdin: "inherit",
  stdout: "inherit",
  stderr: "inherit",
};

const run = async () => {
  Bun.spawn(["bun", "run", "tailwind", "--watch"], spawnOptions);
  // Bun.spawn(["bun", "run", "compile", "--watch"], spawnOptions);
  Bun.spawn(["bun", "run", "start:hot"], spawnOptions);

  process.on("SIGINT", async () => {
    console.log("Cleaning up...");
  });
};

run();
